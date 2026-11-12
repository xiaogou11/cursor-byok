package forwarder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"cursor/gen/agentv1"
	modeladapter "cursor/internal/backend/agent/model"
)

// resumeProbeCompiler 记录 driveProvider 是否被续写触发；Compile 返回固定错误，
// 让被续起的 provider pass 在不依赖真实 provider 的情况下确定性收口。
type resumeProbeCompiler struct {
	compileCalls int
}

var errResumeProbe = errors.New("resume probe: provider pass started")

func (compiler *resumeProbeCompiler) Compile(*ConversationFile, agentv1.AgentMode, string, string, string) (CompiledConversation, error) {
	compiler.compileCalls++
	return CompiledConversation{}, errResumeProbe
}

func (compiler *resumeProbeCompiler) DerivePromptContexts(*ConversationFile, agentv1.AgentMode, string) ([]PromptContextMessage, error) {
	return nil, nil
}

func testMidStreamInterruptedError(cause string) error {
	return providerTerminalError{cause: fmt.Errorf("upstream stream interrupted mid-response: %w: %w", modeladapter.ErrMidStreamInterrupted, errors.New(cause))}
}

func TestInterruptedProviderPassRecoverable(t *testing.T) {
	cases := []struct {
		name string
		pass interruptedProviderPass
		want bool
	}{
		{"idle timeout after partial text", interruptedProviderPass{Cause: testMidStreamInterruptedError("provider stream idle timeout after 90s without effective content")}, true},
		{"chunk read timeout after partial text", interruptedProviderPass{Cause: testMidStreamInterruptedError("read tcp: i/o timeout")}, true},
		{"tool invocation already emitted", interruptedProviderPass{Cause: testMidStreamInterruptedError("read tcp: i/o timeout"), HadToolInvocation: true}, false},
		{"user canceled", interruptedProviderPass{Cause: fmt.Errorf("%w: %w", modeladapter.ErrMidStreamInterrupted, context.Canceled)}, false},
		{"pre-output failure", interruptedProviderPass{Cause: providerTerminalError{cause: errors.New("status=500")}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.pass.recoverable(); got != tc.want {
				t.Fatalf("recoverable() = %t, want %t", got, tc.want)
			}
		})
	}
}

type midStreamFixture struct {
	store    *ConversationFileStore
	service  *Service
	stream   *ActiveStream
	compiler *resumeProbeCompiler
}

func newMidStreamFixture(t *testing.T, requestID string) midStreamFixture {
	t.Helper()
	store := NewConversationFileStore(t.TempDir())
	conversation := testConversation([]HistoryEntry{
		testUserMessageEntry(t, 1, requestID, "长时间执行的任务"),
	})
	persisted, err := store.SaveConversationWithEntries(conversation.ConversationID, conversation, conversation.Entries)
	if err != nil {
		t.Fatalf("SaveConversationWithEntries() error = %v", err)
	}
	broker := NewStreamBroker()
	compiler := &resumeProbeCompiler{}
	service := newServiceWithDependencies(store, NewHistoryProjector(), compiler, nil, broker)
	stream, err := broker.OpenStream(requestID, persisted.ConversationID, 1, "model-a", "model-a", agentv1.AgentMode_AGENT_MODE_AGENT, "长时间执行的任务")
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	stream.CheckpointConversation = cloneConversationFile(persisted)
	stream.CurrentModelCallID = "call-" + requestID
	stream.ProviderPassCount = 1
	stream.Status = StreamStatusStreaming
	return midStreamFixture{store: store, service: service, stream: stream, compiler: compiler}
}

func (fixture midStreamFixture) interruptAfterText(t *testing.T, text string, cause string) {
	t.Helper()
	if err := fixture.service.applyProviderModelEvent(fixture.stream, modeladapter.ModelEvent{Kind: modeladapter.ModelEventKindTextDelta, Text: text}); err != nil {
		t.Fatalf("apply text delta: %v", err)
	}
	if err := fixture.service.handleProviderDoneEvent(fixture.stream, &streamProviderEvent{Done: true, Err: testMidStreamInterruptedError(cause)}); err != nil {
		t.Fatalf("handleProviderDoneEvent() error = %v", err)
	}
}

// entryCounts 统计 canonical 历史中的部分输出条目与续写提示条目。
func (fixture midStreamFixture) entryCounts(t *testing.T) (assistantText int, recoveryReminders int) {
	t.Helper()
	persisted, err := fixture.store.LoadConversation(fixture.stream.ConversationID)
	if err != nil {
		t.Fatalf("LoadConversation() error = %v", err)
	}
	for _, entry := range persisted.Entries {
		switch entry.Kind {
		case "assistant_text":
			assistantText++
		case "prompt_context":
			var payload promptContextEntryPayload
			if err := json.Unmarshal(entry.Payload, &payload); err != nil {
				t.Fatalf("unmarshal prompt_context payload: %v", err)
			}
			if payload.Source == promptContextSourceMidStreamRecovery {
				recoveryReminders++
			}
		}
	}
	return assistantText, recoveryReminders
}

// TestMidStreamInterruptionResumesWithinTurn 锁定核心语义：已输出部分正文后流中断，
// 部分正文落盘一次、追加一条续写提示，并在同一回合内续起下一轮 provider pass。
func TestMidStreamInterruptionResumesWithinTurn(t *testing.T) {
	fixture := newMidStreamFixture(t, "request-mid-stream")
	fixture.interruptAfterText(t, "正在修改第一个文件，", "read tcp: i/o timeout")

	if fixture.compiler.compileCalls != 1 {
		t.Fatalf("resumed provider passes = %d, want 1", fixture.compiler.compileCalls)
	}
	if fixture.stream.MidStreamRecoveryAttempts != 1 {
		t.Fatalf("MidStreamRecoveryAttempts = %d, want 1", fixture.stream.MidStreamRecoveryAttempts)
	}
	assistantText, reminders := fixture.entryCounts(t)
	if assistantText != 1 {
		t.Fatalf("partial assistant_text entries = %d, want 1", assistantText)
	}
	if reminders != 1 {
		t.Fatalf("mid_stream_recovery reminders = %d, want 1", reminders)
	}
}

// TestMidStreamInterruptionExhaustedFallsBackToFailure 锁定连续额度耗尽后回到原失败路径，
// 且部分正文只落盘一次（恢复器未接管时不得提前 flush）。
func TestMidStreamInterruptionExhaustedFallsBackToFailure(t *testing.T) {
	fixture := newMidStreamFixture(t, "request-mid-stream-exhausted")
	fixture.stream.MidStreamRecoveryAttempts = midStreamRecoveryMaxConsecutive
	fixture.interruptAfterText(t, "部分回复。", "provider stream idle timeout after 90s without effective content")

	if fixture.compiler.compileCalls != 0 {
		t.Fatalf("resumed provider passes = %d, want 0 after exhaustion", fixture.compiler.compileCalls)
	}
	if fixture.stream.Status != StreamStatusFailed {
		t.Fatalf("stream status = %q, want %q", fixture.stream.Status, StreamStatusFailed)
	}
	assistantText, reminders := fixture.entryCounts(t)
	if assistantText != 1 {
		t.Fatalf("partial assistant_text entries = %d, want 1", assistantText)
	}
	if reminders != 0 {
		t.Fatalf("mid_stream_recovery reminders = %d, want 0", reminders)
	}
}

// TestSuccessfulProviderPassResetsMidStreamStreak 锁定「连续」语义：成功收口的 pass 清零计数，
// 长任务中偶发断流不会累计耗尽续写额度。
func TestSuccessfulProviderPassResetsMidStreamStreak(t *testing.T) {
	fixture := newMidStreamFixture(t, "request-mid-stream-reset")
	fixture.stream.MidStreamRecoveryAttempts = 2
	if err := fixture.service.applyProviderModelEvent(fixture.stream, modeladapter.ModelEvent{Kind: modeladapter.ModelEventKindTextDelta, Text: "完整回复。"}); err != nil {
		t.Fatalf("apply text delta: %v", err)
	}
	if err := fixture.service.applyProviderModelEvent(fixture.stream, modeladapter.ModelEvent{
		Kind:         modeladapter.ModelEventKindTurnFinished,
		FinishReason: "stop",
		InputTokens:  100,
		OutputTokens: 20,
		UsagePresent: true,
	}); err != nil {
		t.Fatalf("apply turn finished: %v", err)
	}
	if err := fixture.service.handleProviderDoneEvent(fixture.stream, &streamProviderEvent{Done: true}); err != nil {
		t.Fatalf("handleProviderDoneEvent() error = %v", err)
	}
	if fixture.stream.MidStreamRecoveryAttempts != 0 {
		t.Fatalf("MidStreamRecoveryAttempts = %d, want 0 after successful pass", fixture.stream.MidStreamRecoveryAttempts)
	}
}

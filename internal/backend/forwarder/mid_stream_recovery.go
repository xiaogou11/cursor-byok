// mid_stream_recovery.go 处理「上游已向客户端输出部分内容后流中断」的回合级自动续写。
//
// router 层对已产出内容的流（静默超时 / 逐块读超时 / 连接重置）一律不重试：同一请求
// 重发必然重复输出。过去整轮直接以 stream_interrupted 失败，长任务只能靠用户手动「继续」。
// 这里在 forwarder 回合层接管，语义等价于用户点继续，但不结束回合：
//
//	落盘已推送的部分输出 -> 记录失败 pass 用量 -> 追加续写提示 -> resume 下一轮 provider pass
//
// 边界：
//   - 本 pass 已发出完整工具调用时不接管：工具可能已派发执行，续写会重复副作用；
//   - 用户取消不接管；
//   - 连续中断上限 midStreamRecoveryMaxConsecutive；任一 provider pass 成功收口即清零，
//     长任务中偶发断流不会耗尽额度，上游持续不可用时仍能及时失败。
package forwarder

import (
	"context"
	"errors"
	"strings"
	"time"

	"cursor/internal/apperror"
	modeladapter "cursor/internal/backend/agent/model"
	"cursor/internal/logger"
)

const midStreamRecoveryMaxConsecutive = 3

// interruptedProviderPass 描述一次以错误收口的 provider pass，是续写决策的唯一输入。
type interruptedProviderPass struct {
	Cause             error
	Usage             turnUsageSnapshot
	HadToolInvocation bool
	// FlushPartial 把本 pass 已推送给客户端的部分输出写入 canonical 历史。
	FlushPartial func() error
}

func (pass interruptedProviderPass) recoverable() bool {
	return !pass.HadToolInvocation &&
		errors.Is(pass.Cause, modeladapter.ErrMidStreamInterrupted) &&
		!errors.Is(pass.Cause, context.Canceled)
}

// recoverFromMidStreamInterruption 返回 (true, nil) 表示已挂起续写，调用方直接 return；
// (false, nil) 表示不适用或连续额度已耗尽，调用方走原失败路径。
// 返回 error 时部分输出可能已落盘，调用方不得再次 flush，直接失败收口即可。
func (service *Service) recoverFromMidStreamInterruption(stream *ActiveStream, pass interruptedProviderPass) (bool, error) {
	if service == nil || stream == nil || !pass.recoverable() {
		return false, nil
	}
	stream.mu.Lock()
	if stream.MidStreamRecoveryAttempts >= midStreamRecoveryMaxConsecutive {
		attempts := stream.MidStreamRecoveryAttempts
		requestID := stream.RequestID
		stream.mu.Unlock()
		logger.Infof("forwarder mid-stream recovery exhausted request_id=%s consecutive=%d",
			strings.TrimSpace(requestID), attempts)
		return false, nil
	}
	stream.MidStreamRecoveryAttempts++
	attempt := stream.MidStreamRecoveryAttempts
	conversationID := stream.ConversationID
	requestID := stream.RequestID
	turnSeq := stream.TurnSeq
	modelCallID := strings.TrimSpace(stream.CurrentModelCallID)
	stream.UpdatedAt = time.Now().UTC()
	stream.mu.Unlock()

	causeText := strings.TrimSpace(pass.Cause.Error())
	usage := pass.Usage
	if strings.TrimSpace(usage.ErrorCode) == "" {
		usage.ErrorCode = string(apperror.CodeStreamInterrupted)
	}
	steps := []func() error{
		pass.FlushPartial,
		func() error {
			return service.recordTurnUsageSnapshot(stream, conversationID, turnSeq, requestID, modelCallID, "provider_error", usage, causeText, false)
		},
		func() error {
			_, err := service.appendConversationEntries(stream, conversationID, []HistoryEntry{
				newPromptContextEntry(turnSeq, requestID, newPromptContextReminder(promptContextSourceMidStreamRecovery, midStreamRecoveryText())),
			})
			return err
		},
		func() error { return service.syncSummaryCarryForward(conversationID, requestID, modelCallID) },
		func() error { return service.publishCheckpoint(requestID, conversationID) },
	}
	for _, step := range steps {
		if step == nil {
			continue
		}
		if err := step(); err != nil {
			return false, err
		}
	}

	if service.debug != nil {
		service.debug.LogRuntime(context.Background(), requestID, conversationID, "mid_stream_recovery_triggered", map[string]any{
			"model_call_id":   modelCallID,
			"attempt":         attempt,
			"max_consecutive": midStreamRecoveryMaxConsecutive,
			"cause":           causeText,
		})
	}
	logger.Infof("forwarder mid-stream recovery resume request_id=%s attempt=%d/%d cause=%s",
		strings.TrimSpace(requestID), attempt, midStreamRecoveryMaxConsecutive, causeText)

	if err := service.requestProviderAction(stream, providerActionResume); err != nil {
		return false, err
	}
	return true, nil
}

// resetMidStreamRecoveryStreak 在 provider pass 成功收口时调用：连续中断计数只统计「连续」失败。
func resetMidStreamRecoveryStreak(stream *ActiveStream) {
	if stream == nil {
		return
	}
	stream.mu.Lock()
	stream.MidStreamRecoveryAttempts = 0
	stream.mu.Unlock()
}

func midStreamRecoveryText() string {
	return "上一次回复在传输途中被中断（上游流超时或连接断开），并非你主动结束。若上方已有你未说完的回复，用户已经看到，请从中断处直接接着完成本轮任务，不要重复或改写已输出的内容；若中断前正准备发起工具调用，请重新发起该调用。"
}

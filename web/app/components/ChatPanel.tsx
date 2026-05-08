"use client";

import { useEffect, useRef, useState } from "react";
import { frameUrl } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";

interface Props {
  sceneId: string;
  messages: ChatMessage[];
  onSend: (text: string) => void;
  disabled?: boolean;
}

export function ChatPanel({ sceneId, messages, onSend, disabled }: Props) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const submit = () => {
    const t = draft.trim();
    if (!t || disabled) return;
    onSend(t);
    setDraft("");
  };

  return (
    <div className="lp-chat-shell">
      <div ref={scrollRef} className="lp-chat-feed">
        {messages.map((m) => (
          <Message key={m.id} m={m} sceneId={sceneId} />
        ))}
      </div>
      <div className="lp-chat-input--shell">
        <input
          type="text"
          inputMode="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder={disabled ? "Waiting for segmentation…" : "Ask about the scene…"}
          disabled={disabled}
          className="lp-chat-input--field"
        />
        <button
          onClick={submit}
          disabled={disabled || !draft.trim()}
          className="lp-chat-input--send"
        >
          <span className="lp-chat-input--send-glyph">↵</span>
          <span>Send</span>
        </button>
      </div>
    </div>
  );
}

// Some agent replies start with a short editorial preamble (e.g. "Looking…",
// "📍 …"). We split off the first sentence as a serif italic accent so each
// agent message has the editorial moment the design system calls for.
function splitSerifIntro(text: string): { intro: string | null; rest: string } {
  const trimmed = text.trim();
  if (!trimmed) return { intro: null, rest: "" };
  const m = trimmed.match(/^([^.!?\n]{2,40}[.!?])\s+(.+)/s);
  if (!m) return { intro: null, rest: trimmed };
  return { intro: m[1], rest: m[2] };
}

function Message({ m, sceneId }: { m: ChatMessage; sceneId: string }) {
  const isUser = m.role === "user";
  const frames = m.frames_used ?? [];
  const { intro, rest } = isUser ? { intro: null, rest: m.text } : splitSerifIntro(m.text);
  return (
    <div
      className={[
        "flex flex-col animate-slide-in",
        isUser ? "items-end" : "items-start",
      ].join(" ")}
    >
      <div
        className={[
          "lp-bubble",
          isUser ? "lp-bubble--user" : "lp-bubble--agent",
          m.pending ? "lp-bubble-pending" : "",
        ].join(" ")}
      >
        <div className="lp-bubble-text">
          {intro && <span className="lp-bubble-serif">{intro}</span>}
          {rest}
        </div>
        {!isUser && frames.length > 0 && (
          <>
            <div className="lp-bubble-meta">
              looked at {frames.length} frame{frames.length === 1 ? "" : "s"}
            </div>
            <div className="lp-bubble-frames">
              {frames.map((name) => (
                <img
                  key={name}
                  src={frameUrl(sceneId, name)}
                  alt={name}
                  className="lp-bubble-frame"
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

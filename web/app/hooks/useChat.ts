import { useCallback, useState } from "react";
import { postChat } from "@/lib/api";
import type { ChatMessage, Vec3 } from "@/lib/types";
import { useUI } from "@/store/ui";

let nextId = 0;
const newId = () => `msg_${++nextId}_${Date.now().toString(36)}`;

export function useChat(sceneId: string) {
  const camera = useUI((s) => s.camera);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: newId(),
      role: "agent",
      text:
        "Loaded. Ask anything about the scene — try **what's beside the side chair?** to have me look at the frames around an object.",
    },
  ]);

  const send = useCallback(
    async (text: string) => {
      const userMsg: ChatMessage = { id: newId(), role: "user", text };
      const pendingId = newId();
      setMessages((m) => [
        ...m,
        userMsg,
        { id: pendingId, role: "agent", text: "…", pending: true },
      ]);
      try {
        const resp = await postChat({
          scene_id: sceneId,
          message: text,
          camera_pos: camera.position as Vec3,
        });
        setMessages((m) =>
          m.map((msg) =>
            msg.id === pendingId
              ? {
                  ...msg,
                  text: resp.text,
                  pending: false,
                  frames_used: resp.frames_used ?? [],
                  tools_called: resp.tools_called ?? [],
                }
              : msg,
          ),
        );
      } catch (e) {
        const err = e instanceof Error ? e.message : String(e);
        setMessages((m) =>
          m.map((msg) =>
            msg.id === pendingId
              ? { ...msg, text: `(error: ${err})`, pending: false }
              : msg,
          ),
        );
      }
    },
    [sceneId, camera.position],
  );

  return { messages, send };
}

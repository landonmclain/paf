import { MessageContent } from "./message-content";
import { cn } from "@/lib/utils";
import type { User } from "@/lib/db/schema";

export function Message({ author, text, mine }: { author: User; text: string; mine: boolean }) {
  return (
    <div className={cn("message", mine && "message--mine")}>
      <strong>{author.email}</strong>
      <MessageContent text={text} />
    </div>
  );
}

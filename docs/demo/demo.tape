# Regenerate with: vhs docs/demo/demo.tape && bash docs/demo/compress.sh
# (needs vhs, ttyd, ffmpeg and a key in .env)
Output docs/demo/demo.gif

Set Shell bash
Set FontSize 16
Set Width 1200
Set Height 760
Set Padding 20
Set TypingSpeed 40ms
Set PlaybackSpeed 1.5
Set Theme "Catppuccin Mocha"

Hide
Type "export PAF_HOME=$(bash docs/demo/setup.sh) PYDANTIC_AI_NO_BANNER=1 PS1='$ '"
Enter
Type "clear"
Enter
Show

# 1. Tools: streamed tool calls, then a short answer.
Type "paf chat --new"
Enter
Wait@30s /❯/
Type "Find the three largest Python files under src/ and say in one line what each does."
Enter
Sleep 6s
Wait@120s /Message paf/
Sleep 2s

# 2. Approval policy: deletes ask first.
Type "Run rm -r build-cache to clear the build cache."
Enter
Wait@60s /approve\? \[y\/N\]/
Sleep 2s
Type "y"
Enter
Sleep 6s
Wait@60s /Message paf/
Sleep 2s

# 3. Slash commands.
Type "/name Repo tour"
Enter
Sleep 1s
Type "/history 1"
Enter
Sleep 3s
Type "/help"
Enter
Sleep 4s

# 4. Durability: leave, come back, pick up where we stopped.
Type "/quit"
Enter
Sleep 1s
Type "paf chat --resume Repo"
Enter
Wait@30s /❯/
Type "/list"
Enter
Sleep 3s
Type "Which file was the largest?"
Enter
Sleep 6s
Wait@60s /Message paf/
Sleep 3s
Type "/quit"
Enter
Sleep 1s

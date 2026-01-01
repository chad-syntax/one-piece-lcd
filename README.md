# one-piece-lcd

The purpose of this project is to effectively accomplish watching a one piece episode on the lcd screens of my AIO and case fans in my gaming pc.

There are 8 screens total and my thinking was that each one would correspond to a member of the straw hat crew, luffy being dedicated to the AIO since it's the biggest and most prominent.

However it seemed lame to just have a single gif/mp4 of each straw hat on repeat, as their characters develop and are so animated over time I wouldn't be able to choose limited gifs for each.
Then I thought about just taking a whole episode and playing it on these screens with each lcd for one straw hat, but in most shots there are only a few characters and they might not always be straw hats on screen.
So the goal is to create a series of 8 mp4 files (one for each screen) that are 1x1 squares of a characters face like a punch-out. As the episode progresses, whichever character(s) are on screen will be shown on the lcd screens in a hierarchy so that if Luffy, Zorro, and Sanji are on screen, they show up on screens 0, 1, and 2 respectively. If there are no straw hats on screen it will take whatever characters, rank them, and display them.
This way I could effectively play a whole episode if I can take an episode, turn it into 8 separate mp4s, each mp4 would be 60s 400x400 of whatever character in real time.

Limitations

- Lian Li AIO and LCD fans don't have a public api
- Videos are limited to 20mb and it's best to use a 400x400 ish resolution

TODOs

- [ ] download episode(s) in video format
- [ ] gather dataset for one piece characters and their headshots (needed for facial recognition)
- [ ] run facial recognition on episode to get a list of segments, each segment has start/end timestamps with list of present characters and their coordinates
- [ ] take character data and run a series of ffmpeg commands to cut 1:1 segments of the video that is cropped to each persons face
- [ ] combine this video cuts into a series of 8 final videos for my 8 lcd screens
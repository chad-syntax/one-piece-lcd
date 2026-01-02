// Face Detection Viewer JavaScript
// VIDEO_ID and VIDEO_FPS are injected by the HTML template

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const timeline = document.getElementById('timeline');
const timelineMarker = document.getElementById('timeline-marker');

let detections = null;
let frameMap = {};
let frameSkip = 5;
let fps = VIDEO_FPS;
let totalVideoFrames = 0;
let lastDrawnFrame = -1; // Track last drawn frame to avoid redundant redraws

// Pixels per frame in timeline
const PX_PER_FRAME = 4;

// Character colors (consistent per character)
const colors = [
    '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181',
    '#aa96da', '#fcbad3', '#a8d8ea', '#ffd3b6', '#c9b1ff',
    '#ff9f43', '#54a0ff', '#5f27cd', '#00d2d3', '#ff6b81'
];
const charColors = {};
let colorIndex = 0;

function getCharColor(charId) {
    if (charId === 'unknown') return '#666';
    if (!charColors[charId]) {
        charColors[charId] = colors[colorIndex % colors.length];
        colorIndex++;
    }
    return charColors[charId];
}

// Build timeline visualization
function buildTimeline() {
    if (!detections) return;
    
    const timelineLabels = document.getElementById('timeline-labels');
    const timelineInner = document.getElementById('timeline-inner');
    const playhead = document.getElementById('timeline-marker');
    
    totalVideoFrames = detections.total_frames * frameSkip; // Approximate total frames
    const timelineWidth = totalVideoFrames * PX_PER_FRAME / frameSkip;
    
    timelineInner.style.width = `${timelineWidth}px`;
    timelineLabels.innerHTML = '';
    timelineInner.innerHTML = '';
    
    // Re-add the playhead (it was removed by innerHTML clear)
    timelineInner.appendChild(playhead);
    
    // Group frames by character presence
    const characterRanges = {}; // char_id -> [{start, end}]
    
    let lastFrameChars = new Set();
    let rangeStarts = {};
    
    // Sort frame data by frame number
    const sortedFrames = [...detections.frame_data].sort((a, b) => a.frame_number - b.frame_number);
    
    for (const frame of sortedFrames) {
        const frameChars = new Set();
        
        // Add characters from face detections
        for (const face of (frame.faces || [])) {
            if (face.candidates && face.candidates.length > 0) {
                frameChars.add(face.candidates[0].character_id);
            }
        }
        
        // Add characters from full-frame matching
        for (const char of (frame.characters || [])) {
            frameChars.add(char.character_id);
        }
        
        // Check for characters that ended
        for (const char of lastFrameChars) {
            if (!frameChars.has(char) && rangeStarts[char] !== undefined) {
                if (!characterRanges[char]) characterRanges[char] = [];
                characterRanges[char].push({
                    start: rangeStarts[char],
                    end: frame.frame_number
                });
                delete rangeStarts[char];
            }
        }
        
        // Check for characters that started
        for (const char of frameChars) {
            if (rangeStarts[char] === undefined) {
                rangeStarts[char] = frame.frame_number;
            }
        }
        
        lastFrameChars = frameChars;
    }
    
    // Close any open ranges
    const lastFrame = sortedFrames.length > 0 ? sortedFrames[sortedFrames.length - 1].frame_number : 0;
    for (const [char, start] of Object.entries(rangeStarts)) {
        if (!characterRanges[char]) characterRanges[char] = [];
        characterRanges[char].push({ start, end: lastFrame + frameSkip });
    }
    
    // Sort characters by total screen time (most present first)
    const charScreenTime = {};
    for (const [char, ranges] of Object.entries(characterRanges)) {
        charScreenTime[char] = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    }
    const chars = Object.keys(characterRanges).sort((a, b) => charScreenTime[b] - charScreenTime[a]);
    
    for (const char of chars) {
        // Create label in left column
        const label = document.createElement('div');
        label.className = 'timeline-label';
        label.textContent = char.replace(/_/g, ' ');
        label.style.color = getCharColor(char);
        timelineLabels.appendChild(label);
        
        // Create track in scrollable area
        const track = document.createElement('div');
        track.className = 'timeline-track';
        track.style.width = `${timelineWidth}px`;
        track.dataset.character = char;
        
        for (const range of characterRanges[char]) {
            const block = document.createElement('div');
            block.className = 'timeline-block';
            block.style.left = `${range.start * PX_PER_FRAME / frameSkip}px`;
            block.style.width = `${(range.end - range.start) * PX_PER_FRAME / frameSkip}px`;
            block.style.backgroundColor = getCharColor(char);
            block.title = `${char}: frames ${range.start}-${range.end}`;
            track.appendChild(block);
        }
        
        timelineInner.appendChild(track);
    }
}

// Load detections JSON
fetch(`/api/video/${VIDEO_ID}/detections`)
    .then(r => r.json())
    .then(data => {
        detections = data;
        frameSkip = data.config.frame_skip;
        
        // Build frame lookup map (stores full frame data including faces and characters)
        for (const frame of data.frame_data) {
            frameMap[frame.frame_number] = {
                faces: frame.faces || [],
                characters: frame.characters || []
            };
        }
        
        // Show detected characters
        const allChars = document.getElementById('all-chars');
        allChars.innerHTML = data.config.detected_characters
            .map(c => `<span class="char" style="border-left: 3px solid ${getCharColor(c)}">${c.replace(/_/g, ' ')}</span>`)
            .join('');
        
        console.log('Loaded', Object.keys(frameMap).length, 'frames with detections');
        
        // Build timeline after loading
        buildTimeline();
    })
    .catch(err => console.error('Failed to load detections:', err));

// Resize canvas to match video
function resizeCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

video.addEventListener('loadedmetadata', resizeCanvas);
video.addEventListener('resize', resizeCanvas);

// Draw bounding boxes (only redraws when frame changes)
function drawBoxes(force = false) {
    if (!detections) return;
    
    const currentTime = video.currentTime;
    const currentFrame = Math.floor(currentTime * fps / frameSkip) * frameSkip;
    
    // Skip redraw if same frame (unless forced, e.g., on seek)
    if (!force && currentFrame === lastDrawnFrame) return;
    lastDrawnFrame = currentFrame;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    document.getElementById('frame-num').textContent = currentFrame;
    document.getElementById('time').textContent = currentTime.toFixed(2);
    
    // Update timeline marker position
    if (timelineMarker && totalVideoFrames > 0) {
        const markerPos = currentFrame * PX_PER_FRAME / frameSkip;
        timelineMarker.style.left = `${markerPos}px`;
        
        // Auto-scroll timeline to keep marker visible
        const timelineScroll = document.getElementById('timeline-scroll');
        const scrollLeft = timelineScroll.scrollLeft;
        const scrollWidth = timelineScroll.clientWidth;
        
        if (markerPos < scrollLeft + 100 || markerPos > scrollLeft + scrollWidth - 100) {
            timelineScroll.scrollLeft = markerPos - scrollWidth / 2;
        }
    }
    
    const frameData = frameMap[currentFrame] || { faces: [], characters: [] };
    const faces = frameData.faces || [];
    const characters = frameData.characters || [];
    
    if (faces.length === 0 && characters.length === 0) {
        document.getElementById('current-chars').textContent = '-';
        return;
    }
    
    // Collect all detected characters
    const currentChars = [];
    
    // From face detections
    for (const f of faces) {
        if (f.candidates && f.candidates.length > 0) {
            currentChars.push(f.candidates[0].character_id.replace(/_/g, ' ') + ' (face)');
        }
    }
    
    // From full-frame matching
    for (const c of characters) {
        currentChars.push(c.character_id.replace(/_/g, ' ') + ' (frame)');
    }
    
    document.getElementById('current-chars').textContent = currentChars.join(', ') || '-';
    
    // Scale factor (canvas might be different size than displayed video)
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    // Draw face bounding boxes
    for (const face of faces) {
        const { top, left, width, height } = face.coordinates;
        const bestMatch = face.candidates ? face.candidates[0] : null;
        const color = getCharColor(bestMatch ? bestMatch.character_id : 'unknown');
        
        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(left * scaleX, top * scaleY, width * scaleX, height * scaleY);
        
        // Draw label background
        const charId = bestMatch ? bestMatch.character_id : 'unknown';
        const confidence = bestMatch ? bestMatch.confidence : 0;
        const label = `${charId.replace(/_/g, ' ')} (${(confidence * 100).toFixed(0)}%)`;
        ctx.font = 'bold 14px sans-serif';
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(left * scaleX, top * scaleY - 22, textWidth + 10, 22);
        
        // Draw label text
        ctx.fillStyle = charId === 'unknown' ? '#fff' : '#000';
        ctx.fillText(label, left * scaleX + 5, top * scaleY - 6);
        
        // Draw additional candidates below (smaller)
        if (face.candidates && face.candidates.length > 1) {
            ctx.font = '11px sans-serif';
            let yOffset = top * scaleY + height * scaleY + 14;
            
            for (let i = 1; i < Math.min(face.candidates.length, 4); i++) {
                const alt = face.candidates[i];
                const altLabel = `${alt.character_id.replace(/_/g, ' ')} (${(alt.confidence * 100).toFixed(0)}%)`;
                const altColor = getCharColor(alt.character_id);
                
                ctx.fillStyle = 'rgba(0,0,0,0.7)';
                const altWidth = ctx.measureText(altLabel).width;
                ctx.fillRect(left * scaleX, yOffset - 11, altWidth + 6, 14);
                
                ctx.fillStyle = altColor;
                ctx.fillText(altLabel, left * scaleX + 3, yOffset);
                yOffset += 14;
            }
        }
    }
    
    // Display detection summary overlay in top-left corner
    const hasFaces = faces.length > 0;
    const hasChars = characters.length > 0;
    
    if (hasFaces || hasChars) {
        const maxFaces = Math.min(faces.length, 3);
        const maxChars = Math.min(characters.length, 3);
        
        // Calculate actual content height
        let contentHeight = 0;
        if (hasFaces) contentHeight += 16 + maxFaces * 13;
        if (hasFaces && hasChars) contentHeight += 8; // spacing
        if (hasChars) contentHeight += 16 + maxChars * 13;
        
        const boxHeight = contentHeight + 16; // padding
        
        // Draw background
        ctx.fillStyle = 'rgba(0,0,0,0.8)';
        ctx.fillRect(8, 8, 180, boxHeight);
        
        let y = 22;
        
        // Face matches section
        if (hasFaces) {
            ctx.font = 'bold 11px sans-serif';
            ctx.fillStyle = '#4ecdc4';
            ctx.fillText(`Face (${faces.length})`, 12, y);
            y += 14;
            
            ctx.font = '10px sans-serif';
            for (const face of faces.slice(0, maxFaces)) {
                const bestMatch = face.candidates ? face.candidates[0] : null;
                if (bestMatch) {
                    const color = getCharColor(bestMatch.character_id);
                    ctx.fillStyle = color;
                    const name = bestMatch.character_id.replace(/_/g, ' ');
                    ctx.fillText(`${name} ${(bestMatch.confidence * 100).toFixed(0)}%`, 14, y);
                    y += 12;
                }
            }
            if (hasChars) y += 6;
        }
        
        // Full-frame matches section
        if (hasChars) {
            ctx.font = 'bold 11px sans-serif';
            ctx.fillStyle = '#ff9f43';
            ctx.fillText(`Frame (${characters.length})`, 12, y);
            y += 14;
            
            ctx.font = '10px sans-serif';
            for (const char of characters.slice(0, maxChars)) {
                const color = getCharColor(char.character_id);
                ctx.fillStyle = color;
                const name = char.character_id.replace(/_/g, ' ');
                ctx.fillText(`${name} ${(char.confidence * 100).toFixed(0)}%`, 14, y);
                y += 12;
            }
        }
    }
}

// Continuous animation loop - runs at 60fps but only redraws when frame changes
function animate() {
    drawBoxes();
    requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

// Force redraw on seek (in case same frame but user expects visual update)
video.addEventListener('seeked', () => drawBoxes(true));

// Click on timeline to seek
document.getElementById('timeline-scroll').addEventListener('click', (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scrollLeft = e.currentTarget.scrollLeft;
    const clickX = e.clientX - rect.left + scrollLeft;
    
    const frameNum = (clickX / PX_PER_FRAME) * frameSkip;
    const seekTime = frameNum / fps;
    
    video.currentTime = seekTime;
});

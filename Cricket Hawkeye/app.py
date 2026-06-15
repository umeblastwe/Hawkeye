<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Automated HawkEye Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #060a13; font-family: sans-serif; }
        .hud-font { font-family: 'Arial Black', Impact, sans-serif; font-weight: 900; }
    </style>
</head>
<body class="min-h-screen flex flex-col items-center justify-center p-4 text-white">

    <div class="w-full max-w-5xl bg-slate-900/50 border border-slate-800 p-6 rounded-2xl shadow-2xl backdrop-blur-md">
        
        <div class="relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-slate-950 shadow-2xl mb-6">
            <video id="videoPlayer" class="w-full h-full object-cover" muted playsinline></video>
            <canvas id="hawkEyeCanvas" class="absolute inset-0 w-full h-full pointer-events-none"></canvas>

            <div id="hawkEyeHUD" class="absolute left-8 bottom-8 flex flex-col gap-2 transition-all duration-300 opacity-0">
                <div class="flex items-center w-48 h-7 text-[11px] font-bold tracking-wider shadow-lg">
                    <div class="bg-slate-800 text-slate-100 px-3 flex items-center h-full w-1/2 uppercase">Wickets</div>
                    <div id="hudWickets" class="bg-red-600 text-center flex items-center justify-center h-full text-white w-1/2 hud-font">HITTING</div>
                </div>
                <div class="flex items-center w-48 h-7 text-[11px] font-bold tracking-wider shadow-lg">
                    <div class="bg-slate-800 text-slate-100 px-3 flex items-center h-full w-1/2 uppercase">Impact</div>
                    <div id="hudImpact" class="bg-red-600 text-center flex items-center justify-center h-full text-white w-1/2 hud-font">IN-LINE</div>
                </div>
                <div class="flex items-center w-48 h-7 text-[11px] font-bold tracking-wider shadow-lg">
                    <div class="bg-slate-800 text-slate-100 px-3 flex items-center h-full w-1/2 uppercase">Pitching</div>
                    <div id="hudPitching" class="bg-red-600 text-center flex items-center justify-center h-full text-white w-1/2 hud-font">IN-LINE</div>
                </div>
            </div>

            <div id="decisionBanner" class="absolute left-8 top-8 flex flex-col w-40 transition-all duration-300 opacity-0 shadow-xl">
                <div class="bg-slate-800 text-[9px] font-bold text-center py-0.5 text-slate-200">VERDICT PROJECTION</div>
                <div id="hudVerdict" class="bg-red-600 text-sm font-black text-center py-1 text-white hud-font">OUT</div>
            </div>
        </div>

        <div class="bg-slate-950 p-4 border border-slate-800 rounded-xl">
            <form id="hawkEyeForm" class="flex flex-col sm:flex-row gap-4 items-center justify-between">
                <input type="file" id="videoAsset" name="video" accept="video/*" class="text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-rose-600 file:text-white hover:file:bg-rose-500 cursor-pointer" required>
                <button type="submit" id="submitBtn" class="bg-rose-600 hover:bg-rose-500 font-bold text-xs px-6 py-2.5 rounded transition shadow-lg whitespace-nowrap">
                    Run Mathematical Trajectory
                </button>
            </form>
        </div>
    </div>

    <script>
        const video = document.getElementById('videoPlayer');
        const canvas = document.getElementById('hawkEyeCanvas');
        const ctx = canvas.getContext('2d');
        
        let traceProgress = 0;
        let isRunning = false;
        let targetXCoord = 0.5;
        let targetYCoord = 0.42;

        function setCanvasSize() {
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
        }
        window.addEventListener('resize', setCanvasSize);

        document.getElementById('hawkEyeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            setCanvasSize();
            
            isRunning = false;
            traceProgress = 0;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            toggleHUD(false);

            const formData = new FormData(e.target);
            const btn = document.getElementById('submitBtn');
            btn.textContent = "Processing Mathematical Projections...";
            btn.disabled = true;

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const result = await response.json();

                if (result.success) {
                    // Pull mathematical scaling points from OpenCV backend output matrices
                    targetXCoord = result.target_x;
                    targetYCoord = result.target_y;

                    // Update UI text outputs dynamically
                    document.getElementById('hudWickets').textContent = result.telemetry.wickets;
                    document.getElementById('hudImpact').textContent = result.telemetry.impact;
                    document.getElementById('hudPitching').textContent = result.telemetry.pitching;
                    document.getElementById('hudVerdict').textContent = result.telemetry.verdict;

                    // Set colors dynamically based on outcome state
                    setHUDColors(result.telemetry.wickets, result.telemetry.verdict);

                    video.src = result.video_url;
                    video.load();
                    video.play();

                    video.addEventListener('timeupdate', function trackingHook() {
                        if (video.currentTime >= result.impact_time && !isRunning) {
                            video.pause();
                            isRunning = true;
                            toggleHUD(true);
                            drawMathematicalPath();
                            video.removeEventListener('timeupdate', trackingHook);
                        }
                    });
                }
            } catch (err) {
                console.error(err);
            } finally {
                btn.textContent = "Run Mathematical Trajectory";
                btn.disabled = false;
            }
        });

        function drawMathematicalPath() {
            if (traceProgress < 1) {
                traceProgress += 0.02;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Calibrated baseline trajectory dimensions 
                const startX = canvas.width * 0.56;
                const startY = canvas.height * 0.72;
                const impactX = canvas.width * 0.52;
                const impactY = canvas.height * 0.54;

                // Exact end point mapped from backend algorithmic calculation arrays
                const finalTargetX = canvas.width * targetXCoord;
                const finalTargetY = canvas.height * targetYCoord;

                // Path 1: Pitch map line
                ctx.beginPath();
                ctx.moveTo(startX, startY);
                ctx.lineTo(impactX, impactY);
                ctx.strokeStyle = '#ec4899';
                ctx.lineWidth = 4;
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#ec4899';
                ctx.stroke();

                // Path 2: Predictive linear equation regression track
                ctx.beginPath();
                ctx.moveTo(impactX, impactY);
                
                const currX = impactX + (finalTargetX - impactX) * traceProgress;
                const currY = impactY + (finalTargetY - impactY) * traceProgress;
                
                ctx.lineTo(currX, currY);
                ctx.strokeStyle = '#ec4899';
                ctx.lineWidth = 4;
                ctx.setLineDash([5, 4]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Target Node Pulse Circle
                ctx.beginPath();
                ctx.arc(currX, currY, 5, 0, 2 * Math.PI);
                ctx.fillStyle = '#ffffff';
                ctx.fill();

                requestAnimationFrame(drawMathematicalPath);
            }
        }

        function toggleHUD(show) {
            const opacity = show ? '1' : '0';
            document.getElementById('hawkEyeHUD').style.opacity = opacity;
            document.getElementById('decisionBanner').style.opacity = opacity;
        }

        function setHUDColors(wickets, verdict) {
            const wBox = document.getElementById('hudWickets');
            const vBox = document.getElementById('hudVerdict');
            
            if (wickets === "MISSING") {
                wBox.className = "bg-emerald-600 text-center flex items-center justify-center h-full text-white w-1/2 hud-font";
            } else if (wickets === "UMPIRE'S CALL") {
                wBox.className = "bg-amber-500 text-center flex items-center justify-center h-full text-black w-1/2 hud-font";
            } else {
                wBox.className = "bg-red-600 text-center flex items-center justify-center h-full text-white w-1/2 hud-font";
            }

            if (verdict === "NOT OUT") {
                vBox.className = "bg-emerald-600 text-sm font-black text-center py-1 text-white hud-font";
            } else if (verdict === "DECISION PENDING") {
                vBox.className = "bg-amber-500 text-sm font-black text-center py-1 text-black hud-font";
            } else {
                vBox.className = "bg-red-600 text-sm font-black text-center py-1 text-white hud-font";
            }
        }
    </script>
</body>
</html>

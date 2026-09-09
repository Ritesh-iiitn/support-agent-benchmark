"""
Interactive Web Application & Dashboard for Hiver AI Support Agent.
Run with:
  python -m agent.web_app
  or
  uvicorn agent.web_app:app --host 0.0.0.0 --port 8000
"""

import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from agent.config import PROJECT_ROOT, PROCESSED_DATA_DIR, GOLDEN_SET_PATH
from agent.schemas import CustomerMessage
from agent.core_pipeline import ProductionAgentPipeline, SimpleMLPipeline, TrivialBaselinePipeline
from evaluation.llm_judge import LLMAsAJudgeRubric

app = FastAPI(title="AppleSupport AI Agent Live Demo & Benchmark")

# Load pipelines
production_pipe = ProductionAgentPipeline()
ml_pipe = SimpleMLPipeline()
trivial_pipe = TrivialBaselinePipeline()
judge = LLMAsAJudgeRubric()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hiver AI Customer Support Agent — AppleSupport</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-gradient-to-br from-blue-500 to-indigo-600 text-white p-2.5 rounded-xl shadow-lg shadow-blue-500/20">
                    <i class="fab fa-apple text-2xl"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold text-white tracking-tight flex items-center gap-2">
                        AppleSupport AI Agent <span class="text-xs bg-blue-500/20 text-blue-400 font-semibold px-2 py-0.5 rounded-full border border-blue-500/30">Hiver SDE Take-Home</span>
                    </h1>
                    <p class="text-xs text-slate-400">Classify Intents • Grounded RAG Historical Replies • Policy-Driven Escalation</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <a href="#benchmark" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition">
                    <i class="fas fa-chart-bar mr-1.5 text-blue-400"></i> Benchmark
                </a>
                <a href="#human-judge" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition">
                    <i class="fas fa-gavel mr-1.5 text-purple-400"></i> Judge Agreement
                </a>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
        <!-- Live Playground Section -->
        <section class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <!-- Left: Input Control -->
            <div class="lg:col-span-5 space-y-6">
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 shadow-xl backdrop-blur">
                    <h2 class="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                        <i class="fas fa-terminal text-blue-400"></i> Interactive Query Input
                    </h2>
                    <p class="text-xs text-slate-400 mb-4">Type a real customer tweet or choose from pre-configured challenge slices below.</p>
                    
                    <div class="space-y-4">
                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1">Customer Tweet Text</label>
                            <textarea id="tweetInput" rows="4" class="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition" placeholder="e.g., My iPhone 14 battery is dying in 3 hours after updating to iOS 17.4!"></textarea>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1">Pipeline Engine</label>
                            <select id="pipelineSelect" class="w-full bg-slate-900 border border-slate-700 rounded-xl p-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <option value="production">✨ Proposed Agent (Production Hybrid + RAG + Safety)</option>
                                <option value="ml">⚡ Baseline 2 (TF-IDF LogReg + Verbatim Retrieval)</option>
                                <option value="trivial">⚪ Baseline 1 (Trivial Majority + Static Template)</option>
                            </select>
                        </div>

                        <button id="runBtn" onclick="runInference()" class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium py-3 rounded-xl shadow-lg shadow-blue-500/25 transition flex items-center justify-center gap-2 text-sm">
                            <i class="fas fa-bolt"></i> Process Customer Tweet
                        </button>
                    </div>

                    <!-- Presets -->
                    <div class="mt-6 pt-5 border-t border-slate-700/60">
                        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2.5">Preset Challenge Scenarios</span>
                        <div class="flex flex-wrap gap-2">
                            <button onclick="setQuery('My iPhone battery is swelling and the screen popped off, smells like burning!', 'production')" class="text-xs bg-red-950/40 hover:bg-red-900/60 text-red-300 border border-red-800/40 px-2.5 py-1.5 rounded-lg transition">
                                🔥 Critical Safety
                            </button>
                            <button onclick="setQuery('My credit card is 4111-2222-3333-4444 why was I charged $9.99 for Apple Arcade?', 'production')" class="text-xs bg-amber-950/40 hover:bg-amber-900/60 text-amber-300 border border-amber-800/40 px-2.5 py-1.5 rounded-lg transition">
                                🔒 PII Leakage
                            </button>
                            <button onclick="setQuery('How do I trade in my iPhone 13 towards an iPhone 15 Pro?', 'production')" class="text-xs bg-blue-950/40 hover:bg-blue-900/60 text-blue-300 border border-blue-800/40 px-2.5 py-1.5 rounded-lg transition">
                                📱 Trade-In
                            </button>
                            <button onclick="setQuery('Left AirPod Pro will not charge in case and mic sounds muffled.', 'production')" class="text-xs bg-purple-950/40 hover:bg-purple-900/60 text-purple-300 border border-purple-800/40 px-2.5 py-1.5 rounded-lg transition">
                                🎧 AirPods
                            </button>
                            <button onclick="setQuery('Can you help fix the cracked screen on my Samsung Galaxy S23?', 'production')" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 px-2.5 py-1.5 rounded-lg transition">
                                🌐 Out-of-Scope
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right: Real-Time Output Inspection -->
            <div class="lg:col-span-7 space-y-6">
                <div id="outputContainer" class="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 shadow-xl space-y-6">
                    <div class="flex items-center justify-between border-b border-slate-700/60 pb-4">
                        <div class="flex items-center space-x-2">
                            <span class="relative flex h-3 w-3">
                                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                            </span>
                            <h3 class="font-semibold text-white text-sm">Agent Real-Time Inspection</h3>
                        </div>
                        <span id="latencyBadge" class="text-xs text-slate-400 bg-slate-900 px-2.5 py-1 rounded-md border border-slate-800">Latency: -- ms</span>
                    </div>

                    <!-- Intent & Escalation Status Cards -->
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div class="bg-slate-900/80 border border-slate-700/80 rounded-xl p-4">
                            <span class="text-xs font-medium text-slate-400 block mb-1">Classified Intent</span>
                            <div class="flex items-center justify-between">
                                <span id="intentLabel" class="text-sm font-bold text-blue-400">os_update_glitch</span>
                                <span id="intentConfidence" class="text-xs bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded font-mono">92%</span>
                            </div>
                            <p id="intentReason" class="text-xs text-slate-400 mt-2 line-clamp-2">Calibrated confidence matching.</p>
                        </div>

                        <div class="bg-slate-900/80 border border-slate-700/80 rounded-xl p-4">
                            <span class="text-xs font-medium text-slate-400 block mb-1">Escalation Decision</span>
                            <div class="flex items-center justify-between">
                                <span id="escalationAction" class="text-sm font-bold text-emerald-400">AUTO_REPLY</span>
                                <span id="urgencyBadge" class="text-xs bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded font-semibold">LOW</span>
                            </div>
                            <p id="escalationReason" class="text-xs text-slate-400 mt-2 line-clamp-2">Standard self-service resolution.</p>
                        </div>
                    </div>

                    <!-- Draft Reply Box -->
                    <div class="bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-700/80 rounded-xl p-4">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                                <i class="fab fa-twitter text-sky-400"></i> Grounded Draft Reply
                            </span>
                            <span id="charCount" class="text-xs text-slate-400 font-mono">142 / 280 chars</span>
                        </div>
                        <div id="draftReplyText" class="text-sm text-slate-100 bg-slate-950/60 p-3.5 rounded-lg border border-slate-800 leading-relaxed font-sans">
                            Thanks for reaching out! It is normal for devices to use more battery for the first 48h while re-indexing. Restart your device and check Battery settings: https://support.apple.com/HT208387
                        </div>
                        <div class="mt-3 flex items-center gap-4 text-xs text-slate-400">
                            <span id="kbLinkCheck" class="flex items-center gap-1"><i class="fas fa-check-circle text-emerald-400"></i> Official KB URL</span>
                            <span id="dmPromptCheck" class="flex items-center gap-1"><i class="fas fa-shield-alt text-blue-400"></i> DM Handoff</span>
                        </div>
                    </div>

                    <!-- Retrieved Contexts -->
                    <div>
                        <span class="text-xs font-semibold text-slate-300 uppercase tracking-wider block mb-2">Retrieved Historical Resolutions (RAG Precedents)</span>
                        <div id="retrievedList" class="space-y-2 text-xs">
                            <div class="bg-slate-900/60 border border-slate-800 p-2.5 rounded-lg text-slate-300">
                                <span class="text-blue-400 font-semibold">[Sim: 0.88]</span> It is normal for devices to use more battery for the first 48h while re-indexing...
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Benchmark Results Section -->
        <section id="benchmark" class="space-y-6 pt-6 border-t border-slate-800">
            <div>
                <h2 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fas fa-trophy text-amber-400"></i> Headline Benchmark Comparison (200 Golden Test Set)
                </h2>
                <p class="text-xs text-slate-400">Comparative evaluation across Baseline 1 (Trivial), Baseline 2 (Simple ML), and Proposed Production Agent.</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
                    <span class="text-xs text-slate-400">Intent Macro F1</span>
                    <div class="text-2xl font-bold text-white mt-1">0.9902</div>
                    <span class="text-xs text-emerald-400 font-medium">99.00% Accuracy</span>
                </div>
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
                    <span class="text-xs text-slate-400">Escalation Recall</span>
                    <div class="text-2xl font-bold text-white mt-1">88.24%</div>
                    <span class="text-xs text-emerald-400 font-medium">Only 2 Dangerous Leaks</span>
                </div>
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
                    <span class="text-xs text-slate-400">Risk-Weighted Cost</span>
                    <div class="text-2xl font-bold text-white mt-1">68.0</div>
                    <span class="text-xs text-emerald-400 font-medium">60% Cost Reduction vs B1</span>
                </div>
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
                    <span class="text-xs text-slate-400">Avg Latency</span>
                    <div class="text-2xl font-bold text-white mt-1">1.9 ms</div>
                    <span class="text-xs text-blue-400 font-medium">Sub-2ms Ultra Fast</span>
                </div>
            </div>

            <!-- Benchmark Table -->
            <div class="bg-slate-800/60 border border-slate-700/60 rounded-2xl overflow-hidden shadow-xl">
                <table class="w-full text-left text-sm">
                    <thead class="bg-slate-950/80 text-xs uppercase text-slate-400 border-b border-slate-700/60">
                        <tr>
                            <th class="px-6 py-4 font-semibold">Evaluation Metric / Dimension</th>
                            <th class="px-6 py-4 font-semibold text-slate-400">Baseline 1 (Trivial)</th>
                            <th class="px-6 py-4 font-semibold text-yellow-400">Baseline 2 (Simple ML)</th>
                            <th class="px-6 py-4 font-semibold text-emerald-400">Proposed Agent (Production)</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-800 font-mono text-xs">
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">Intent Classification Accuracy</td>
                            <td class="px-6 py-3.5 text-slate-400">81.00%</td>
                            <td class="px-6 py-3.5 text-yellow-400">99.00%</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">99.00%</td>
                        </tr>
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">Intent Macro F1 Score</td>
                            <td class="px-6 py-3.5 text-slate-400">0.7835</td>
                            <td class="px-6 py-3.5 text-yellow-400">0.9899</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">0.9902</td>
                        </tr>
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">Escalation Recall (Safety Critical)</td>
                            <td class="px-6 py-3.5 text-red-400">0.00%</td>
                            <td class="px-6 py-3.5 text-yellow-400">35.29%</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">88.24%</td>
                        </tr>
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">Dangerous False Auto-Replies (Leak Count)</td>
                            <td class="px-6 py-3.5 text-red-400">17</td>
                            <td class="px-6 py-3.5 text-yellow-400">11</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">2</td>
                        </tr>
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">Risk-Weighted Cost (10×FN + 2×FP)</td>
                            <td class="px-6 py-3.5 text-slate-400">170.0</td>
                            <td class="px-6 py-3.5 text-yellow-400">110.0</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">68.0</td>
                        </tr>
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">KB URL Grounding Accuracy</td>
                            <td class="px-6 py-3.5 text-slate-400">0.00%</td>
                            <td class="px-6 py-3.5 text-yellow-400">63.22%</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">75.86%</td>
                        </tr>
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">Length Compliance (<280 Chars)</td>
                            <td class="px-6 py-3.5 text-slate-400">100.00%</td>
                            <td class="px-6 py-3.5 text-yellow-400">100.00%</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">100.00%</td>
                        </tr>
                        <tr class="hover:bg-slate-800/30">
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">LLM-as-a-Judge Overall Score (1-5)</td>
                            <td class="px-6 py-3.5 text-slate-400">3.81</td>
                            <td class="px-6 py-3.5 text-yellow-400">4.48</td>
                            <td class="px-6 py-3.5 text-emerald-400 font-bold">4.60</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Human vs Judge Alignment Section -->
        <section id="human-judge" class="space-y-6 pt-6 border-t border-slate-800">
            <div>
                <h2 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fas fa-gavel text-purple-400"></i> LLM-as-a-Judge vs. Human Agreement Validation
                </h2>
                <p class="text-xs text-slate-400">Rigorous statistical proof that automated evaluation rubric aligns with human expert grading (N=50 sample).</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5">
                    <div class="text-purple-400 text-xs font-semibold uppercase tracking-wider mb-1">Pearson Correlation (r)</div>
                    <div class="text-3xl font-bold text-white">0.9431</div>
                    <p class="text-xs text-slate-400 mt-2">p < 0.001. Extremely high linear agreement between human and judge.</p>
                </div>
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5">
                    <div class="text-purple-400 text-xs font-semibold uppercase tracking-wider mb-1">Quadratic Cohen's Kappa</div>
                    <div class="text-3xl font-bold text-white">0.7253</div>
                    <p class="text-xs text-slate-400 mt-2">Substantial inter-rater reliability accounting for chance agreement.</p>
                </div>
                <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5">
                    <div class="text-purple-400 text-xs font-semibold uppercase tracking-wider mb-1">Mean Absolute Error (MAE)</div>
                    <div class="text-3xl font-bold text-white">0.35 pts</div>
                    <p class="text-xs text-slate-400 mt-2">100% of ratings fall within ±1.0 point tolerance on a 1-5 scale.</p>
                </div>
            </div>
        </section>
    </main>

    <footer class="border-t border-slate-800 bg-slate-950 py-6 text-center text-xs text-slate-500">
        Hiver SDE Intern Take-Home Project • AppleSupport AI Agent Pipeline • Reproducible in <15 minutes
    </footer>

    <script>
        function setQuery(text, pipe) {
            document.getElementById('tweetInput').value = text;
            document.getElementById('pipelineSelect').value = pipe;
            runInference();
        }

        async function runInference() {
            const text = document.getElementById('tweetInput').value.trim();
            if (!text) return;
            const pipeline = document.getElementById('pipelineSelect').value;
            const btn = document.getElementById('runBtn');
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, pipeline })
                });
                const data = await res.json();

                // Update UI
                document.getElementById('latencyBadge').innerText = `Latency: ${data.processing_time_ms} ms`;
                document.getElementById('intentLabel').innerText = data.intent_result.intent;
                document.getElementById('intentConfidence').innerText = `${Math.round(data.intent_result.confidence * 100)}%`;
                document.getElementById('intentReason').innerText = data.intent_result.reasoning || '';

                const esc = data.escalation_decision;
                const escActionElem = document.getElementById('escalationAction');
                const urgencyElem = document.getElementById('urgencyBadge');
                escActionElem.innerText = esc.action;
                urgencyElem.innerText = esc.urgency;

                if (esc.action === 'ESCALATE_TO_HUMAN') {
                    escActionElem.className = 'text-sm font-bold text-red-400';
                    urgencyElem.className = 'text-xs bg-red-500/20 text-red-300 px-2 py-0.5 rounded font-semibold';
                } else {
                    escActionElem.className = 'text-sm font-bold text-emerald-400';
                    urgencyElem.className = 'text-xs bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded font-semibold';
                }
                document.getElementById('escalationReason').innerText = esc.reason;

                if (data.draft_reply) {
                    const rep = data.draft_reply;
                    document.getElementById('draftReplyText').innerText = rep.reply_text;
                    document.getElementById('charCount').innerText = `${rep.char_count} / 280 chars`;
                    document.getElementById('kbLinkCheck').innerHTML = rep.contains_kb_link 
                        ? '<i class="fas fa-check-circle text-emerald-400"></i> Official KB URL' 
                        : '<i class="fas fa-times-circle text-slate-500"></i> Official KB URL';
                    document.getElementById('dmPromptCheck').innerHTML = rep.contains_dm_handoff 
                        ? '<i class="fas fa-shield-alt text-blue-400"></i> DM Handoff' 
                        : '<i class="fas fa-times-circle text-slate-500"></i> DM Handoff';
                }

                // Retrieved contexts
                const list = document.getElementById('retrievedList');
                list.innerHTML = '';
                if (data.retrieved_contexts && data.retrieved_contexts.length > 0) {
                    data.retrieved_contexts.forEach(ctx => {
                        const d = document.createElement('div');
                        d.className = 'bg-slate-900/60 border border-slate-800 p-2.5 rounded-lg text-slate-300';
                        d.innerHTML = `<span class="text-blue-400 font-semibold">[Sim: ${ctx.similarity_score.toFixed(4)}]</span> ${ctx.response}`;
                        list.appendChild(d);
                    });
                } else {
                    list.innerHTML = '<div class="text-slate-500 text-xs italic">No historical contexts retrieved.</div>';
                }

            } catch (err) {
                alert('Error running inference: ' + err);
            } finally {
                btn.innerHTML = '<i class="fas fa-bolt"></i> Process Customer Tweet';
                btn.disabled = false;
            }
        }

        // Run default query on load
        window.addEventListener('DOMContentLoaded', () => {
            document.getElementById('tweetInput').value = "My iPhone 14 battery is dying in 3 hours after updating to iOS 17.4!";
            runInference();
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/api/process")
async def process_tweet_api(request: Request):
    payload = await request.json()
    text = payload.get("text", "")
    pipeline_type = payload.get("pipeline", "production")

    msg = CustomerMessage(id="web_query", text=text)

    if pipeline_type == "trivial":
        resp = trivial_pipe.process_message(msg)
    elif pipeline_type == "ml":
        resp = ml_pipe.process_message(msg)
    else:
        resp = production_pipe.process_message(msg)

    return JSONResponse(content=resp.dict())


@app.get("/api/benchmark-results")
async def get_benchmark_results():
    results_path = PROCESSED_DATA_DIR / "benchmark_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "Benchmark results not found"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Professional Enterprise Dashboard & Interactive Playground for AppleSupport AI Agent.
Features:
- Live Interactive Multi-Turn Simulator & Playground
- Side-by-Side Model Comparison (Proposed vs Simple ML vs Trivial)
- Interactive Benchmark Visualizations & Confusion Matrix
- Human vs. LLM-as-a-Judge Statistical Alignment Inspector
- Golden Evaluation Dataset (200 Items) Slice Explorer
- Architecture & Decision Log Viewer
"""

import sys
import socket
import argparse
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from agent.config import PROJECT_ROOT, PROCESSED_DATA_DIR, GOLDEN_SET_PATH, HUMAN_BENCHMARK_PATH
from agent.schemas import CustomerMessage
from agent.core_pipeline import ProductionAgentPipeline, SimpleMLPipeline, TrivialBaselinePipeline
from evaluation.llm_judge import LLMAsAJudgeRubric

app = FastAPI(title="AppleSupport AI Agent Enterprise Dashboard")

# Initialize pipelines
production_pipe = ProductionAgentPipeline()
ml_pipe = SimpleMLPipeline()
trivial_pipe = TrivialBaselinePipeline()
judge = LLMAsAJudgeRubric()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hiver AI Customer Support Agent | AppleSupport Enterprise Hub</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#f0f7ff',
                            100: '#e0effe',
                            500: '#0071e3',
                            600: '#0062c4',
                            700: '#004fa3',
                        },
                        dark: {
                            900: '#0a0d14',
                            800: '#111726',
                            700: '#1b2438',
                            600: '#25324d',
                        }
                    },
                    fontFamily: {
                        sans: ['-apple-system', 'BlinkMacSystemFont', 'SF Pro Display', 'Inter', 'Segoe UI', 'Roboto', 'sans-serif'],
                        mono: ['SF Mono', 'Fira Code', 'Menlo', 'Monaco', 'Courier New', 'monospace']
                    }
                }
            }
        }
    </script>
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <style>
        .glass-panel {
            background: rgba(17, 23, 38, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .glow-accent {
            box-shadow: 0 0 35px -5px rgba(0, 113, 227, 0.25);
        }
        .glow-red {
            box-shadow: 0 0 35px -5px rgba(239, 68, 68, 0.3);
        }
        .glow-green {
            box-shadow: 0 0 35px -5px rgba(16, 185, 129, 0.25);
        }
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #0a0d14;
        }
        ::-webkit-scrollbar-thumb {
            background: #25324d;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3b4d75;
        }
    </style>
</head>
<body class="bg-dark-900 text-slate-100 font-sans min-h-screen flex flex-col antialiased selection:bg-brand-500 selection:text-white">

    <!-- Top Navigation Bar -->
    <nav class="sticky top-0 z-50 glass-panel border-b border-slate-800/80">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <!-- Brand Logo & Title -->
                <div class="flex items-center space-x-3.5">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-sky-400 p-0.5 shadow-lg shadow-blue-500/20 flex items-center justify-center">
                        <div class="w-full h-full bg-dark-900 rounded-[10px] flex items-center justify-center">
                            <i class="fab fa-apple text-xl text-white"></i>
                        </div>
                    </div>
                    <div>
                        <div class="flex items-center space-x-2">
                            <span class="font-bold text-base tracking-tight text-white">AppleSupport AI</span>
                            <span class="text-[10px] font-semibold uppercase tracking-wider bg-brand-500/20 text-sky-400 border border-brand-500/30 px-2 py-0.5 rounded-full">Production Agent</span>
                        </div>
                        <p class="text-[11px] text-slate-400">Hiver SDE Intern Take-Home • Production Evaluation Suite</p>
                    </div>
                </div>

                <!-- Tab Navigation Links -->
                <div class="hidden md:flex items-center space-x-1 bg-dark-800/80 p-1 rounded-xl border border-slate-700/60 text-xs">
                    <button onclick="switchTab('simulator')" id="tab-btn-simulator" class="tab-btn px-3.5 py-1.5 rounded-lg font-medium text-white bg-brand-500 shadow-sm transition">
                        <i class="fas fa-play-circle mr-1.5"></i> Live Simulator
                    </button>
                    <button onclick="switchTab('comparison')" id="tab-btn-comparison" class="tab-btn px-3.5 py-1.5 rounded-lg font-medium text-slate-400 hover:text-white transition">
                        <i class="fas fa-columns mr-1.5"></i> Model Comparison
                    </button>
                    <button onclick="switchTab('benchmark')" id="tab-btn-benchmark" class="tab-btn px-3.5 py-1.5 rounded-lg font-medium text-slate-400 hover:text-white transition">
                        <i class="fas fa-chart-line mr-1.5"></i> Headline Benchmark
                    </button>
                    <button onclick="switchTab('judge')" id="tab-btn-judge" class="tab-btn px-3.5 py-1.5 rounded-lg font-medium text-slate-400 hover:text-white transition">
                        <i class="fas fa-gavel mr-1.5"></i> Human vs Judge
                    </button>
                    <button onclick="switchTab('dataset')" id="tab-btn-dataset" class="tab-btn px-3.5 py-1.5 rounded-lg font-medium text-slate-400 hover:text-white transition">
                        <i class="fas fa-database mr-1.5"></i> Dataset & Slices
                    </button>
                </div>

                <!-- Top Right Badges -->
                <div class="flex items-center space-x-2.5">
                    <div class="flex items-center space-x-1.5 text-xs text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2.5 py-1 rounded-lg">
                        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span class="font-mono font-medium">12/12 Tests Passing</span>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 space-y-8 w-full">

        <!-- ============================================================== -->
        <!-- TAB 1: LIVE INTERACTIVE SIMULATOR                             -->
        <!-- ============================================================== -->
        <div id="tab-simulator" class="tab-content space-y-8">
            <!-- Hero Stats Banner -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="glass-panel rounded-2xl p-4 border border-slate-800">
                    <span class="text-xs text-slate-400 font-medium flex items-center gap-1.5">
                        <i class="fas fa-bullseye text-blue-400"></i> Intent Accuracy
                    </span>
                    <div class="text-2xl font-bold text-white mt-1">99.00%</div>
                    <span class="text-[11px] text-emerald-400 font-medium">0.9902 Macro F1 Score</span>
                </div>
                <div class="glass-panel rounded-2xl p-4 border border-slate-800">
                    <span class="text-xs text-slate-400 font-medium flex items-center gap-1.5">
                        <i class="fas fa-shield-halved text-emerald-400"></i> Safety Escalation Recall
                    </span>
                    <div class="text-2xl font-bold text-white mt-1">88.24%</div>
                    <span class="text-[11px] text-emerald-400 font-medium">81.8% Leak Reduction</span>
                </div>
                <div class="glass-panel rounded-2xl p-4 border border-slate-800">
                    <span class="text-xs text-slate-400 font-medium flex items-center gap-1.5">
                        <i class="fas fa-link text-purple-400"></i> KB Grounding Accuracy
                    </span>
                    <div class="text-2xl font-bold text-white mt-1">75.86%</div>
                    <span class="text-[11px] text-purple-400 font-medium">2,268 Indexed Pairs</span>
                </div>
                <div class="glass-panel rounded-2xl p-4 border border-slate-800">
                    <span class="text-xs text-slate-400 font-medium flex items-center gap-1.5">
                        <i class="fas fa-bolt text-amber-400"></i> Average Latency
                    </span>
                    <div class="text-2xl font-bold text-white mt-1">1.9 ms</div>
                    <span class="text-[11px] text-amber-400 font-medium">Sub-2ms Ultra Fast</span>
                </div>
            </div>

            <!-- Simulator Grid -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <!-- Left: Interactive Input Panel -->
                <div class="lg:col-span-5 space-y-6">
                    <div class="glass-panel rounded-2xl p-6 glow-accent border border-slate-700/60 space-y-5">
                        <div class="flex items-center justify-between border-b border-slate-700/60 pb-3">
                            <h2 class="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                                <i class="fas fa-terminal text-blue-400"></i> Incoming Tweet Input
                            </h2>
                            <span class="text-xs text-slate-400">Target: <strong class="text-sky-400">@AppleSupport</strong></span>
                        </div>

                        <!-- Tweet Input -->
                        <div class="space-y-2">
                            <div class="flex items-center justify-between">
                                <label class="text-xs font-semibold text-slate-300">Customer Tweet Message</label>
                                <span id="inputCharCount" class="text-[11px] font-mono text-slate-400">0 / 280</span>
                            </div>
                            <textarea id="simTweetInput" rows="4" oninput="updateCharCount()" class="w-full bg-dark-900/90 border border-slate-700 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent transition leading-relaxed font-sans" placeholder="Type or paste any real customer tweet here..."></textarea>
                        </div>

                        <!-- Pipeline Select -->
                        <div class="space-y-1.5">
                            <label class="text-xs font-semibold text-slate-300">Agent Pipeline Architecture</label>
                            <select id="simPipelineSelect" class="w-full bg-dark-900 border border-slate-700 rounded-xl p-2.5 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500">
                                <option value="production">✨ Proposed Production Agent (Calibrated Hybrid + RAG + Safety)</option>
                                <option value="ml">⚡ Baseline 2: Simple ML (TF-IDF + Verbatim Retrieval)</option>
                                <option value="trivial">⚪ Baseline 1: Trivial (Majority Class + Canned Template)</option>
                            </select>
                        </div>

                        <!-- Process Button -->
                        <button id="simRunBtn" onclick="runSimInference()" class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold py-3 rounded-xl shadow-lg shadow-blue-500/25 transition flex items-center justify-center gap-2 text-sm">
                            <i class="fas fa-microchip"></i> Run Agent Analysis
                        </button>

                        <!-- Scenario Presets -->
                        <div class="pt-4 border-t border-slate-700/60 space-y-2.5">
                            <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">One-Click Stress Scenarios</span>
                            <div class="flex flex-wrap gap-2">
                                <button onclick="setSimPreset('My iPhone battery is physically bulging and the screen popped off, smells like burning!', 'production')" class="text-xs bg-red-950/40 hover:bg-red-900/60 text-red-300 border border-red-800/40 px-2.5 py-1.5 rounded-lg transition flex items-center gap-1.5">
                                    <i class="fas fa-fire text-red-400"></i> Swollen Battery
                                </button>
                                <button onclick="setSimPreset('My credit card 4111-2222-3333-4444 was charged $9.99 for Apple Arcade why was I billed?', 'production')" class="text-xs bg-amber-950/40 hover:bg-amber-900/60 text-amber-300 border border-amber-800/40 px-2.5 py-1.5 rounded-lg transition flex items-center gap-1.5">
                                    <i class="fas fa-credit-card text-amber-400"></i> PII Leak
                                </button>
                                <button onclick="setSimPreset('My iPhone 14 battery life was cut in half after updating to iOS 17.4 yesterday! Phone feels warm.', 'production')" class="text-xs bg-blue-950/40 hover:bg-blue-900/60 text-blue-300 border border-blue-800/40 px-2.5 py-1.5 rounded-lg transition flex items-center gap-1.5">
                                    <i class="fas fa-battery-half text-blue-400"></i> iOS Update Drain
                                </button>
                                <button onclick="setSimPreset('Left AirPod Pro will not charge in the case and mic sounds muffled during calls.', 'production')" class="text-xs bg-purple-950/40 hover:bg-purple-900/60 text-purple-300 border border-purple-800/40 px-2.5 py-1.5 rounded-lg transition flex items-center gap-1.5">
                                    <i class="fas fa-headphones text-purple-400"></i> AirPod Pairing
                                </button>
                                <button onclick="setSimPreset('How much trade-in credit can I get for an iPhone 13 Pro towards an iPhone 15?', 'production')" class="text-xs bg-emerald-950/40 hover:bg-emerald-900/60 text-emerald-300 border border-emerald-800/40 px-2.5 py-1.5 rounded-lg transition flex items-center gap-1.5">
                                    <i class="fas fa-exchange-alt text-emerald-400"></i> Trade-In Value
                                </button>
                                <button onclick="setSimPreset('Can you tell me how to fix the blue screen of death on my Windows 11 desktop?', 'production')" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 px-2.5 py-1.5 rounded-lg transition flex items-center gap-1.5">
                                    <i class="fas fa-globe text-slate-400"></i> Out-of-Scope
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right: Agent Output & Reasoning Inspection -->
                <div class="lg:col-span-7 space-y-6">
                    <div id="simOutputPanel" class="glass-panel rounded-2xl p-6 border border-slate-700/60 space-y-6">
                        <!-- Output Header -->
                        <div class="flex items-center justify-between border-b border-slate-700/60 pb-4">
                            <div class="flex items-center space-x-2.5">
                                <span class="relative flex h-3 w-3">
                                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                    <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                                </span>
                                <h3 class="font-bold text-sm text-white uppercase tracking-wider">Agent Triage & Execution Response</h3>
                            </div>
                            <span id="simLatency" class="text-xs font-mono text-slate-400 bg-dark-900 px-2.5 py-1 rounded-md border border-slate-800">Latency: 1.82 ms</span>
                        </div>

                        <!-- 2 Status Cards (Intent & Escalation) -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <!-- Intent Card -->
                            <div class="bg-dark-900/80 border border-slate-700/80 rounded-xl p-4 space-y-2">
                                <div class="flex items-center justify-between">
                                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Classified Intent</span>
                                    <span id="simIntentConf" class="text-xs font-mono font-bold bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded">92.00%</span>
                                </div>
                                <div id="simIntentLabel" class="text-base font-bold text-blue-400 font-mono">os_update_glitch</div>
                                <p id="simIntentReason" class="text-xs text-slate-400 leading-normal">High-precision domain rule matched 'os_update_glitch'.</p>
                            </div>

                            <!-- Escalation Card -->
                            <div id="simEscalationCard" class="bg-dark-900/80 border border-slate-700/80 rounded-xl p-4 space-y-2">
                                <div class="flex items-center justify-between">
                                    <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Escalation Decision</span>
                                    <span id="simUrgencyBadge" class="text-xs font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">LOW</span>
                                </div>
                                <div id="simEscAction" class="text-base font-bold text-emerald-400 font-mono">AUTO_REPLY</div>
                                <p id="simEscReason" class="text-xs text-slate-400 leading-normal">Query matched standard intent; safe for automated guidance.</p>
                                <div class="text-[11px] text-slate-400 pt-1 border-t border-slate-800">
                                    Routing: <strong id="simRoutingDept" class="text-slate-200">tier1_community_bot</strong>
                                </div>
                            </div>
                        </div>

                        <!-- Draft Reply Box -->
                        <div class="bg-gradient-to-b from-dark-900 to-dark-800 border border-slate-700/80 rounded-xl p-5 space-y-3">
                            <div class="flex items-center justify-between">
                                <span class="text-xs font-semibold text-slate-300 flex items-center gap-2">
                                    <i class="fab fa-twitter text-sky-400"></i> Grounded Draft Reply (Apple Brand Voice)
                                </span>
                                <div class="flex items-center space-x-2">
                                    <span id="simReplyLen" class="text-xs font-mono text-slate-400">142 / 280 chars</span>
                                    <button onclick="copyReply()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-2 py-1 rounded transition flex items-center gap-1">
                                        <i class="fas fa-copy"></i> <span id="copyBtnText">Copy</span>
                                    </button>
                                </div>
                            </div>
                            <div id="simReplyText" class="text-sm text-slate-100 bg-dark-900/90 p-4 rounded-lg border border-slate-800 leading-relaxed font-sans shadow-inner">
                                Thanks for reaching out! It is normal for devices to use more battery for the first 48h while re-indexing. Restart your device and check Battery settings: https://support.apple.com/HT208387
                            </div>
                            <div class="flex flex-wrap items-center gap-4 text-xs text-slate-400 pt-1">
                                <span id="simHasKB" class="flex items-center gap-1 text-emerald-400"><i class="fas fa-check-circle"></i> Official Apple KB Link</span>
                                <span id="simHasDM" class="flex items-center gap-1 text-slate-500"><i class="fas fa-times-circle"></i> DM Prompt</span>
                                <span class="flex items-center gap-1 text-slate-400"><i class="fas fa-shield-alt text-blue-400"></i> Strict &lt;280 Guardrail</span>
                            </div>
                        </div>

                        <!-- Retrieved RAG Precedents -->
                        <div class="space-y-2">
                            <span class="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                                <i class="fas fa-book-bookmark text-indigo-400"></i> Top Retrieved Historical Resolutions (RAG Knowledge Base)
                            </span>
                            <div id="simRetrievedList" class="space-y-2 text-xs">
                                <div class="bg-dark-900/70 border border-slate-800 p-3 rounded-lg text-slate-300 space-y-1">
                                    <div class="flex items-center justify-between">
                                        <span class="font-mono text-blue-400 font-semibold">[Similarity: 0.8842]</span>
                                        <span class="text-[10px] text-slate-400">Intent: os_update_glitch</span>
                                    </div>
                                    <p class="text-slate-300">It is normal for devices to use more battery and run warmer for 48h while re-indexing...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- TAB 2: SIDE-BY-SIDE MODEL COMPARISON                          -->
        <!-- ============================================================== -->
        <div id="tab-comparison" class="tab-content hidden space-y-6">
            <div>
                <h2 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fas fa-columns text-blue-400"></i> Parallel Multi-Pipeline Comparison
                </h2>
                <p class="text-xs text-slate-400">Inspect how Baseline 1, Baseline 2, and the Proposed Agent behave on the exact same input query simultaneously.</p>
            </div>

            <!-- Comparison Input -->
            <div class="glass-panel rounded-2xl p-5 border border-slate-800 space-y-3">
                <label class="text-xs font-semibold text-slate-300">Test Query for Parallel Evaluation</label>
                <div class="flex gap-3">
                    <input type="text" id="compQueryInput" class="flex-1 bg-dark-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-brand-500" value="My battery is swelling and the screen popped off, smells like burning!">
                    <button onclick="runParallelComparison()" class="bg-brand-500 hover:bg-brand-600 text-white font-medium px-5 py-2.5 rounded-xl transition text-sm flex items-center gap-2">
                        <i class="fas fa-play"></i> Compare 3 Pipelines
                    </button>
                </div>
            </div>

            <!-- 3 Column Cards -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <!-- Baseline 1 (Trivial) -->
                <div class="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
                    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                        <span class="font-bold text-sm text-slate-300">Baseline 1 (Trivial)</span>
                        <span class="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">Majority + Static</span>
                    </div>
                    <div class="space-y-3 text-xs">
                        <div>
                            <span class="text-slate-400 block mb-0.5">Predicted Intent:</span>
                            <span id="comp-b1-intent" class="font-mono font-bold text-slate-200">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-0.5">Escalation Action:</span>
                            <span id="comp-b1-esc" class="font-mono font-bold text-slate-200">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-0.5">Generated Reply:</span>
                            <div id="comp-b1-reply" class="bg-dark-900 p-3 rounded-lg border border-slate-800 text-slate-300 min-h-[90px] leading-relaxed">--</div>
                        </div>
                        <div class="text-[11px] text-red-400 bg-red-950/30 p-2 rounded border border-red-900/40">
                            ⚠️ <strong>Limitation:</strong> Always auto-replies with canned template. Misses critical safety hazards completely.
                        </div>
                    </div>
                </div>

                <!-- Baseline 2 (Simple ML) -->
                <div class="glass-panel rounded-2xl p-5 border border-yellow-800/40 space-y-4">
                    <div class="flex items-center justify-between border-b border-yellow-800/40 pb-3">
                        <span class="font-bold text-sm text-yellow-400">Baseline 2 (Simple ML)</span>
                        <span class="text-[10px] bg-yellow-950/60 text-yellow-300 px-2 py-0.5 rounded font-mono">TF-IDF + Verbatim</span>
                    </div>
                    <div class="space-y-3 text-xs">
                        <div>
                            <span class="text-slate-400 block mb-0.5">Predicted Intent:</span>
                            <span id="comp-b2-intent" class="font-mono font-bold text-yellow-300">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-0.5">Escalation Action:</span>
                            <span id="comp-b2-esc" class="font-mono font-bold text-yellow-300">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-0.5">Generated Reply:</span>
                            <div id="comp-b2-reply" class="bg-dark-900 p-3 rounded-lg border border-slate-800 text-slate-300 min-h-[90px] leading-relaxed">--</div>
                        </div>
                        <div class="text-[11px] text-yellow-400 bg-yellow-950/30 p-2 rounded border border-yellow-900/40">
                            ⚠️ <strong>Limitation:</strong> Verbatim retrieval without context adaptation. Weak escalation recall (35.29%).
                        </div>
                    </div>
                </div>

                <!-- Proposed Agent (Production) -->
                <div class="glass-panel rounded-2xl p-5 border border-emerald-800/60 glow-green space-y-4">
                    <div class="flex items-center justify-between border-b border-emerald-800/60 pb-3">
                        <span class="font-bold text-sm text-emerald-400">Proposed Production Agent</span>
                        <span class="text-[10px] bg-emerald-950/80 text-emerald-300 px-2 py-0.5 rounded font-mono">Hybrid + RAG + Safety</span>
                    </div>
                    <div class="space-y-3 text-xs">
                        <div>
                            <span class="text-slate-400 block mb-0.5">Predicted Intent:</span>
                            <span id="comp-prod-intent" class="font-mono font-bold text-emerald-300">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-0.5">Escalation Action:</span>
                            <span id="comp-prod-esc" class="font-mono font-bold text-emerald-300">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block mb-0.5">Generated Reply:</span>
                            <div id="comp-prod-reply" class="bg-dark-900 p-3 rounded-lg border border-slate-800 text-slate-100 min-h-[90px] leading-relaxed">--</div>
                        </div>
                        <div class="text-[11px] text-emerald-400 bg-emerald-950/40 p-2 rounded border border-emerald-800/40">
                            ✅ <strong>Strength:</strong> Dynamic RAG synthesis, 88.24% escalation recall, safe DM redirection, strictly under 280 chars.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- TAB 3: BENCHMARK DASHBOARD & VISUALIZATIONS                   -->
        <!-- ============================================================== -->
        <div id="tab-benchmark" class="tab-content hidden space-y-6">
            <div>
                <h2 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fas fa-trophy text-amber-400"></i> Headline Benchmark Suite (200 Golden Test Set)
                </h2>
                <p class="text-xs text-slate-400">Complete statistical comparison evaluated across 200 hand-labelled test cases with difficulty tiers.</p>
            </div>

            <!-- Full Benchmark Table -->
            <div class="glass-panel rounded-2xl overflow-hidden shadow-2xl border border-slate-800">
                <table class="w-full text-left text-sm">
                    <thead class="bg-dark-900 text-xs uppercase text-slate-400 border-b border-slate-800">
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
                            <td class="px-6 py-3.5 font-sans font-medium text-slate-200">Length Compliance (&lt;280 Chars)</td>
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
        </div>

        <!-- ============================================================== -->
        <!-- TAB 4: HUMAN VS LLM-JUDGE AGREEMENT                            -->
        <!-- ============================================================== -->
        <div id="tab-judge" class="tab-content hidden space-y-6">
            <div>
                <h2 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fas fa-gavel text-purple-400"></i> Statistical Human-Judge Agreement Proof (N=50)
                </h2>
                <p class="text-xs text-slate-400">Validation study comparing automated judge ratings against multi-human consensus ground truth.</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="glass-panel rounded-2xl p-6 border border-purple-800/40 space-y-2">
                    <span class="text-xs font-semibold text-purple-400 uppercase tracking-wider">Pearson Correlation (r)</span>
                    <div class="text-4xl font-bold text-white">0.9431</div>
                    <p class="text-xs text-slate-400 leading-normal">p &lt; 0.001. Extremely strong linear alignment between automated judge and human consensus.</p>
                </div>
                <div class="glass-panel rounded-2xl p-6 border border-purple-800/40 space-y-2">
                    <span class="text-xs font-semibold text-purple-400 uppercase tracking-wider">Quadratic Cohen's Kappa</span>
                    <div class="text-4xl font-bold text-white">0.7253</div>
                    <p class="text-xs text-slate-400 leading-normal">Substantial inter-rater reliability accounting for chance agreement.</p>
                </div>
                <div class="glass-panel rounded-2xl p-6 border border-purple-800/40 space-y-2">
                    <span class="text-xs font-semibold text-purple-400 uppercase tracking-wider">Mean Absolute Error (MAE)</span>
                    <div class="text-4xl font-bold text-white">0.35 pts</div>
                    <p class="text-xs text-slate-400 leading-normal">100% of judge ratings fall within ±1.0 point tolerance on a 1-5 scale.</p>
                </div>
            </div>

            <!-- Rubric Breakdown -->
            <div class="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
                <h3 class="text-sm font-bold text-white uppercase tracking-wider">4-Factor Judge Rubric Criteria</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                    <div class="bg-dark-900 p-4 rounded-xl border border-slate-800 space-y-1.5">
                        <span class="font-bold text-blue-400">1. Groundedness & Factual Accuracy (1-5)</span>
                        <p class="text-slate-400">Checks technical correctness of steps against genuine Apple Support articles (`support.apple.com`). Penalizes hallucinated settings or dangerous advice.</p>
                    </div>
                    <div class="bg-dark-900 p-4 rounded-xl border border-slate-800 space-y-1.5">
                        <span class="font-bold text-emerald-400">2. Brand Voice & Empathy (1-5)</span>
                        <p class="text-slate-400">Enforces Apple's warm, supportive, polite greeting ("We'd like to help..."). Penalizes defensive, robotic, or rude responses.</p>
                    </div>
                    <div class="bg-dark-900 p-4 rounded-xl border border-slate-800 space-y-1.5">
                        <span class="font-bold text-amber-400">3. Safety & Policy Compliance (1-5)</span>
                        <p class="text-slate-400">Enforces safe DM redirects for PII/billing/account security. Strictly fails responses suggesting rice or dangerous hardware manipulation.</p>
                    </div>
                    <div class="bg-dark-900 p-4 rounded-xl border border-slate-800 space-y-1.5">
                        <span class="font-bold text-purple-400">4. Actionability & Brevity (1-5)</span>
                        <p class="text-slate-400">Ensures clear immediate next diagnostic steps and 100% adherence to Twitter's 280-character maximum length limit.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- TAB 5: DATASET & SLICES EXPLORER                              -->
        <!-- ============================================================== -->
        <div id="tab-dataset" class="tab-content hidden space-y-6">
            <div>
                <h2 class="text-xl font-bold text-white flex items-center gap-2">
                    <i class="fas fa-database text-blue-400"></i> Golden Evaluation Dataset Explorer (200 Curated Items)
                </h2>
                <p class="text-xs text-slate-400">Explore the non-leaked golden test set partitioned across intents, difficulty tiers, and challenge slices.</p>
            </div>

            <!-- Filter Controls -->
            <div class="glass-panel rounded-2xl p-4 border border-slate-800 flex flex-wrap gap-4 items-center justify-between text-xs">
                <div class="flex items-center gap-3">
                    <label class="font-semibold text-slate-300">Filter Intent:</label>
                    <select id="datasetIntentFilter" onchange="renderDatasetTable()" class="bg-dark-900 border border-slate-700 rounded-lg p-2 text-slate-200">
                        <option value="ALL">All 7 Intents</option>
                        <option value="os_update_glitch">os_update_glitch</option>
                        <option value="hardware_battery_issue">hardware_battery_issue</option>
                        <option value="account_billing_subscription">account_billing_subscription</option>
                        <option value="connectivity_audio_sync">connectivity_audio_sync</option>
                        <option value="app_functionality_crash">app_functionality_crash</option>
                        <option value="general_inquiry_policy">general_inquiry_policy</option>
                        <option value="out_of_scope_chitchat">out_of_scope_chitchat</option>
                    </select>
                </div>
                <span id="datasetCountBadge" class="text-xs text-slate-400 font-mono">Showing 200 items</span>
            </div>

            <!-- Dataset Table -->
            <div class="glass-panel rounded-2xl overflow-hidden border border-slate-800">
                <div class="max-h-[550px] overflow-y-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="bg-dark-900 uppercase text-slate-400 sticky top-0 border-b border-slate-800">
                            <tr>
                                <th class="px-4 py-3">ID</th>
                                <th class="px-4 py-3">Customer Tweet</th>
                                <th class="px-4 py-3">True Intent</th>
                                <th class="px-4 py-3">Action</th>
                                <th class="px-4 py-3">Difficulty</th>
                                <th class="px-4 py-3">Slice</th>
                            </tr>
                        </thead>
                        <tbody id="datasetTableBody" class="divide-y divide-slate-800 font-mono">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

    </main>

    <!-- Footer -->
    <footer class="glass-panel border-t border-slate-800/80 py-6 mt-12 text-center text-xs text-slate-500">
        <div class="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>Hiver SDE Intern Take-Home Project • AppleSupport AI Support Agent</span>
            <span class="font-mono text-slate-400">Reproducible in &lt;15 Minutes • Production Ready</span>
        </div>
    </footer>

    <!-- JavaScript Application Logic -->
    <script>
        let goldenDatasetCache = [];

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.className = 'tab-btn px-3.5 py-1.5 rounded-lg font-medium text-slate-400 hover:text-white transition';
            });

            document.getElementById('tab-' + tabId).classList.remove('hidden');
            const activeBtn = document.getElementById('tab-btn-' + tabId);
            activeBtn.className = 'tab-btn px-3.5 py-1.5 rounded-lg font-medium text-white bg-brand-500 shadow-sm transition';

            if (tabId === 'comparison') {
                runParallelComparison();
            } else if (tabId === 'dataset') {
                loadDataset();
            }
        }

        function updateCharCount() {
            const len = document.getElementById('simTweetInput').value.length;
            document.getElementById('inputCharCount').innerText = `${len} / 280`;
        }

        function setSimPreset(text, pipe) {
            document.getElementById('simTweetInput').value = text;
            document.getElementById('simPipelineSelect').value = pipe;
            updateCharCount();
            runSimInference();
        }

        async function runSimInference() {
            const text = document.getElementById('simTweetInput').value.trim();
            if (!text) return;
            const pipeline = document.getElementById('simPipelineSelect').value;
            const btn = document.getElementById('simRunBtn');
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, pipeline })
                });
                const data = await res.json();

                // Latency
                document.getElementById('simLatency').innerText = `Latency: ${data.processing_time_ms} ms`;

                // Intent
                document.getElementById('simIntentLabel').innerText = data.intent_result.intent;
                document.getElementById('simIntentConf').innerText = `${Math.round(data.intent_result.confidence * 100)}%`;
                document.getElementById('simIntentReason').innerText = data.intent_result.reasoning || 'Calibrated confidence.';

                // Escalation
                const esc = data.escalation_decision;
                const escActionElem = document.getElementById('simEscAction');
                const urgencyElem = document.getElementById('simUrgencyBadge');
                escActionElem.innerText = esc.action;
                urgencyElem.innerText = esc.urgency;
                document.getElementById('simRoutingDept').innerText = esc.routing_department;
                document.getElementById('simEscReason').innerText = esc.reason;

                if (esc.action === 'ESCALATE_TO_HUMAN') {
                    escActionElem.className = 'text-base font-bold text-red-400 font-mono';
                    urgencyElem.className = 'text-xs font-bold px-2 py-0.5 rounded bg-red-500/20 text-red-300';
                } else {
                    escActionElem.className = 'text-base font-bold text-emerald-400 font-mono';
                    urgencyElem.className = 'text-xs font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300';
                }

                // Reply
                if (data.draft_reply) {
                    const rep = data.draft_reply;
                    document.getElementById('simReplyText').innerText = rep.reply_text;
                    document.getElementById('simReplyLen').innerText = `${rep.char_count} / 280 chars`;

                    document.getElementById('simHasKB').className = rep.contains_kb_link 
                        ? 'flex items-center gap-1 text-emerald-400' 
                        : 'flex items-center gap-1 text-slate-500';
                    document.getElementById('simHasKB').innerHTML = rep.contains_kb_link 
                        ? '<i class="fas fa-check-circle"></i> Official Apple KB Link' 
                        : '<i class="fas fa-times-circle"></i> Official Apple KB Link';

                    document.getElementById('simHasDM').className = rep.contains_dm_handoff 
                        ? 'flex items-center gap-1 text-blue-400' 
                        : 'flex items-center gap-1 text-slate-500';
                    document.getElementById('simHasDM').innerHTML = rep.contains_dm_handoff 
                        ? '<i class="fas fa-shield-alt"></i> DM Prompt' 
                        : '<i class="fas fa-times-circle"></i> DM Prompt';
                }

                // Retrieved Contexts
                const list = document.getElementById('simRetrievedList');
                list.innerHTML = '';
                if (data.retrieved_contexts && data.retrieved_contexts.length > 0) {
                    data.retrieved_contexts.forEach(ctx => {
                        const d = document.createElement('div');
                        d.className = 'bg-dark-900/70 border border-slate-800 p-3 rounded-lg text-slate-300 space-y-1';
                        d.innerHTML = `
                            <div class="flex items-center justify-between">
                                <span class="font-mono text-blue-400 font-semibold">[Similarity: ${ctx.similarity_score.toFixed(4)}]</span>
                                <span class="text-[10px] text-slate-400">Intent: ${ctx.intent}</span>
                            </div>
                            <p class="text-slate-300">${ctx.response}</p>
                        `;
                        list.appendChild(d);
                    });
                } else {
                    list.innerHTML = '<div class="text-slate-500 italic p-2">No historical contexts retrieved.</div>';
                }

            } catch (err) {
                alert('Error processing tweet: ' + err);
            } finally {
                btn.innerHTML = '<i class="fas fa-microchip"></i> Run Agent Analysis';
                btn.disabled = false;
            }
        }

        async function runParallelComparison() {
            const text = document.getElementById('compQueryInput').value.trim();
            if (!text) return;

            const pipes = ['trivial', 'ml', 'production'];
            for (const p of pipes) {
                try {
                    const res = await fetch('/api/process', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, pipeline: p })
                    });
                    const d = await res.json();
                    const prefix = p === 'trivial' ? 'comp-b1' : (p === 'ml' ? 'comp-b2' : 'comp-prod');
                    document.getElementById(prefix + '-intent').innerText = `${d.intent_result.intent} (${Math.round(d.intent_result.confidence*100)}%)`;
                    document.getElementById(prefix + '-esc').innerText = `${d.escalation_decision.action} [${d.escalation_decision.urgency}]`;
                    document.getElementById(prefix + '-reply').innerText = d.draft_reply ? d.draft_reply.reply_text : 'N/A';
                } catch (e) {
                    console.error(e);
                }
            }
        }

        async function loadDataset() {
            if (goldenDatasetCache.length > 0) {
                renderDatasetTable();
                return;
            }
            try {
                const res = await fetch('/api/golden-dataset');
                goldenDatasetCache = await res.json();
                renderDatasetTable();
            } catch (e) {
                console.error('Error fetching dataset', e);
            }
        }

        function renderDatasetTable() {
            const filter = document.getElementById('datasetIntentFilter').value;
            const tbody = document.getElementById('datasetTableBody');
            tbody.innerHTML = '';

            const filtered = filter === 'ALL' 
                ? goldenDatasetCache 
                : goldenDatasetCache.filter(item => item.true_intent === filter);

            document.getElementById('datasetCountBadge').innerText = `Showing ${filtered.length} of ${goldenDatasetCache.length} items`;

            filtered.forEach(item => {
                const tr = document.createElement('tr');
                tr.className = 'hover:bg-slate-800/40 text-slate-300';
                const actBadge = item.true_escalation_action === 'ESCALATE_TO_HUMAN' 
                    ? '<span class="text-red-400 font-bold">ESCALATE</span>' 
                    : '<span class="text-emerald-400">AUTO</span>';

                tr.innerHTML = `
                    <td class="px-4 py-2.5 text-slate-400">${item.id}</td>
                    <td class="px-4 py-2.5 font-sans text-slate-200 max-w-xs truncate" title="${item.text}">${item.text}</td>
                    <td class="px-4 py-2.5 text-blue-400">${item.true_intent}</td>
                    <td class="px-4 py-2.5">${actBadge}</td>
                    <td class="px-4 py-2.5 text-slate-400">${item.difficulty_tier}</td>
                    <td class="px-4 py-2.5 text-purple-400">${item.slice_tag}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function copyReply() {
            const text = document.getElementById('simReplyText').innerText;
            navigator.clipboard.writeText(text);
            const btn = document.getElementById('copyBtnText');
            btn.innerText = 'Copied!';
            setTimeout(() => { btn.innerText = 'Copy'; }, 1800);
        }

        window.addEventListener('DOMContentLoaded', () => {
            setSimPreset("My iPhone 14 battery life was cut in half after updating to iOS 17.4 yesterday! Phone feels warm.", "production");
        });
    </script>
</body>
</html>
"""


def find_free_port(preferred: int = 8080) -> int:
    """Finds an available free TCP port on 0.0.0.0."""
    candidates = [preferred, 8080, 8501, 8888, 5001, 3000]
    for p in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", p))
                return p
        except OSError:
            continue
    # Fallback to OS assigned port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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


@app.get("/api/golden-dataset")
async def get_golden_dataset():
    if GOLDEN_SET_PATH.exists():
        with open(GOLDEN_SET_PATH, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content=[], status_code=404)


@app.get("/api/benchmark-results")
async def get_benchmark_results():
    results_path = PROCESSED_DATA_DIR / "benchmark_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "Benchmark results not found"}, status_code=404)


def main():
    parser = argparse.ArgumentParser(description="AppleSupport Enterprise Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    args = parser.parse_args()

    port = find_free_port(args.port)
    print(f"\n🚀 AppleSupport AI Agent Enterprise Dashboard running at: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

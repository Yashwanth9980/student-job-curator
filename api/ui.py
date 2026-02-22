"""
api/ui.py
─────────
Serves the browser-facing internship dashboard at GET /.

The page is a single self-contained HTML file that:
  - Fetches jobs from GET /jobs via the browser's fetch() API
  - Computes "No. of Openings" per company client-side
  - Renders a searchable / filterable table with direct apply links
  - Requires no build step – Tailwind CSS is loaded from CDN
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

ui_router = APIRouter(tags=["ui"])

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Bangalore Internships</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    /* Smooth skeleton shimmer */
    @keyframes shimmer {
      0%   { background-position: -800px 0; }
      100% { background-position:  800px 0; }
    }
    .skeleton {
      background: linear-gradient(90deg, #e5e7eb 25%, #f3f4f6 50%, #e5e7eb 75%);
      background-size: 800px 100%;
      animation: shimmer 1.4s infinite;
      border-radius: 4px;
    }
  </style>
</head>
<body class="bg-gray-50 min-h-screen font-sans">

  <!-- ── Header ───────────────────────────────────────────────────────── -->
  <header class="bg-indigo-700 text-white shadow-md">
    <div class="max-w-7xl mx-auto px-4 py-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
      <div>
        <h1 class="text-2xl font-bold tracking-tight">Bangalore Internship Curator</h1>
        <p class="text-indigo-200 text-sm mt-0.5">
          Curated from Greenhouse &amp; Lever · region: Bangalore / India
        </p>
      </div>
      <div class="text-right text-sm text-indigo-200">
        Last refreshed: <span id="refresh-time" class="font-medium text-white">—</span>
      </div>
    </div>
  </header>

  <!-- ── Main content ─────────────────────────────────────────────────── -->
  <main class="max-w-7xl mx-auto px-4 py-8 space-y-6">

    <!-- Stats bar -->
    <div id="stats-bar" class="grid grid-cols-2 sm:grid-cols-3 gap-4">
      <div class="bg-white rounded-xl shadow-sm p-4 border border-gray-100">
        <p class="text-xs text-gray-500 uppercase tracking-wide">Total Openings</p>
        <p id="stat-total" class="text-3xl font-bold text-indigo-700 mt-1">—</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-4 border border-gray-100">
        <p class="text-xs text-gray-500 uppercase tracking-wide">Companies Hiring</p>
        <p id="stat-companies" class="text-3xl font-bold text-indigo-700 mt-1">—</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-4 border border-gray-100 col-span-2 sm:col-span-1">
        <p class="text-xs text-gray-500 uppercase tracking-wide">Platforms</p>
        <p id="stat-platforms" class="text-3xl font-bold text-indigo-700 mt-1">—</p>
      </div>
    </div>

    <!-- Search & filter row -->
    <div class="flex flex-col sm:flex-row gap-3">
      <input
        id="search"
        type="text"
        placeholder="Search company or position…"
        class="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 text-sm shadow-sm
               focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent"
      />
      <select
        id="platform-filter"
        class="rounded-lg border border-gray-300 px-3 py-2.5 text-sm shadow-sm
               focus:outline-none focus:ring-2 focus:ring-indigo-400 bg-white"
      >
        <option value="">All platforms</option>
        <option value="greenhouse">Greenhouse</option>
        <option value="lever">Lever</option>
      </select>
      <button
        id="refresh-btn"
        class="rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2.5 text-sm
               font-medium shadow-sm transition-colors"
        onclick="loadJobs()"
      >
        Refresh
      </button>
    </div>

    <!-- Table card -->
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-x-auto">
      <table class="min-w-full text-sm">
        <thead>
          <tr class="border-b border-gray-100 bg-gray-50">
            <th class="px-5 py-3 text-left font-semibold text-gray-600 w-40">Company</th>
            <th class="px-5 py-3 text-left font-semibold text-gray-600">Intern Position</th>
            <th class="px-5 py-3 text-left font-semibold text-gray-600 w-36">Location</th>
            <th class="px-5 py-3 text-center font-semibold text-gray-600 w-28">No. of Openings</th>
            <th class="px-5 py-3 text-center font-semibold text-gray-600 w-28">Apply</th>
          </tr>
        </thead>
        <tbody id="jobs-tbody">
          <!-- populated by JS -->
        </tbody>
      </table>

      <!-- Loading skeleton -->
      <div id="loading" class="p-6 space-y-3">
        <div class="skeleton h-8 w-full"></div>
        <div class="skeleton h-8 w-5/6"></div>
        <div class="skeleton h-8 w-4/6"></div>
        <div class="skeleton h-8 w-full"></div>
        <div class="skeleton h-8 w-3/4"></div>
      </div>

      <!-- Empty state -->
      <div id="empty-state" class="hidden py-16 text-center text-gray-400">
        <svg class="mx-auto mb-3 h-10 w-10 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z" />
        </svg>
        <p class="font-medium">No internships found</p>
        <p class="text-xs mt-1">Try a different search or run the scraper pipeline first.</p>
      </div>

      <!-- Error state -->
      <div id="error-state" class="hidden py-12 text-center text-red-400">
        <p class="font-medium" id="error-msg">Failed to load jobs.</p>
        <p class="text-xs mt-1">Check that the API server is running and the database is seeded.</p>
      </div>
    </div>

    <!-- Results count -->
    <p id="result-count" class="text-xs text-gray-400 text-right hidden"></p>

  </main>

  <footer class="mt-12 pb-6 text-center text-xs text-gray-400">
    Student Job Curator · Bangalore Internship Edition
  </footer>

  <script>
    // ── State ──────────────────────────────────────────────────────────────
    let allJobs = [];          // full raw list from API
    let companyOpenings = {};  // { company: count } across allJobs

    // ── Fetch ──────────────────────────────────────────────────────────────
    async function loadJobs() {
      showLoading(true);

      try {
        // Fetch up to 200 jobs (max allowed by the API)
        const res = await fetch("/jobs?limit=200&offset=0");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        allJobs = data.jobs || [];

        // Pre-compute per-company openings count
        companyOpenings = {};
        for (const job of allJobs) {
          companyOpenings[job.company] = (companyOpenings[job.company] || 0) + 1;
        }

        updateStats(allJobs);
        renderTable(filtered());
        document.getElementById("refresh-time").textContent =
          new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });

      } catch (err) {
        showError(err.message);
      } finally {
        showLoading(false);
      }
    }

    // ── Filter ─────────────────────────────────────────────────────────────
    function filtered() {
      const q  = document.getElementById("search").value.toLowerCase().trim();
      const pl = document.getElementById("platform-filter").value;

      return allJobs.filter(job => {
        const matchText = !q ||
          job.company.toLowerCase().includes(q) ||
          job.title.toLowerCase().includes(q);
        const matchPlatform = !pl || job.platform === pl;
        return matchText && matchPlatform;
      });
    }

    // ── Stats ──────────────────────────────────────────────────────────────
    function updateStats(jobs) {
      const companies = new Set(jobs.map(j => j.company)).size;
      const platforms = new Set(jobs.map(j => j.platform)).size;
      document.getElementById("stat-total").textContent     = jobs.length;
      document.getElementById("stat-companies").textContent = companies;
      document.getElementById("stat-platforms").textContent = platforms;
    }

    // ── Render table ───────────────────────────────────────────────────────
    function renderTable(jobs) {
      const tbody  = document.getElementById("jobs-tbody");
      const empty  = document.getElementById("empty-state");
      const count  = document.getElementById("result-count");

      tbody.innerHTML = "";

      if (jobs.length === 0) {
        empty.classList.remove("hidden");
        count.classList.add("hidden");
        return;
      }

      empty.classList.add("hidden");

      // Alternate row shading per company group
      let prevCompany = null;
      let shade = false;

      jobs.forEach(job => {
        if (job.company !== prevCompany) {
          shade = !shade;
          prevCompany = job.company;
        }

        const bg = shade ? "bg-white" : "bg-gray-50/60";
        const openings = companyOpenings[job.company] ?? 1;

        // Platform badge colour
        const badgeClass = job.platform === "greenhouse"
          ? "bg-green-100 text-green-700"
          : "bg-purple-100 text-purple-700";

        const tr = document.createElement("tr");
        tr.className = `${bg} border-b border-gray-100 hover:bg-indigo-50/50 transition-colors`;
        tr.innerHTML = `
          <td class="px-5 py-3 font-medium text-gray-800 whitespace-nowrap">
            ${esc(job.company)}
            <span class="ml-1.5 text-[10px] px-1.5 py-0.5 rounded-full font-medium ${badgeClass}">
              ${esc(job.platform)}
            </span>
          </td>
          <td class="px-5 py-3 text-gray-700">${esc(job.title)}</td>
          <td class="px-5 py-3 text-gray-500 text-xs whitespace-nowrap">${esc(job.location || "Bangalore, India")}</td>
          <td class="px-5 py-3 text-center">
            <span class="inline-flex items-center justify-center min-w-[2rem] px-2 py-0.5
                         rounded-full text-xs font-bold bg-indigo-100 text-indigo-700">
              ${openings}
            </span>
          </td>
          <td class="px-5 py-3 text-center">
            <a href="${esc(job.url)}" target="_blank" rel="noopener noreferrer"
               class="inline-block rounded-lg bg-indigo-600 hover:bg-indigo-700
                      text-white text-xs font-medium px-4 py-1.5 transition-colors">
              Apply →
            </a>
          </td>
        `;
        tbody.appendChild(tr);
      });

      count.textContent = `Showing ${jobs.length} result${jobs.length !== 1 ? "s" : ""}`;
      count.classList.remove("hidden");
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    function esc(str) {
      const d = document.createElement("div");
      d.textContent = str ?? "";
      return d.innerHTML;
    }

    function showLoading(on) {
      document.getElementById("loading").classList.toggle("hidden", !on);
      document.getElementById("jobs-tbody").innerHTML = on ? "" : document.getElementById("jobs-tbody").innerHTML;
    }

    function showError(msg) {
      document.getElementById("error-state").classList.remove("hidden");
      document.getElementById("error-msg").textContent = "Error: " + msg;
    }

    // ── Live search & filter ───────────────────────────────────────────────
    document.getElementById("search").addEventListener("input", () => renderTable(filtered()));
    document.getElementById("platform-filter").addEventListener("change", () => renderTable(filtered()));

    // ── Boot ──────────────────────────────────────────────────────────────
    loadJobs();
  </script>
</body>
</html>
"""


@ui_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the internship dashboard SPA."""
    return HTMLResponse(content=_HTML)

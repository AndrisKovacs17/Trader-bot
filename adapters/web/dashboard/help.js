// Context-sensitive help overlay for the KCA-Mamba dashboard.
//
// Shared across every dashboard page. Adds a floating "?" button
// in the bottom-right corner; clicking it opens a modal with the
// help text corresponding to the current URL path.
//
// Activation: include via <script src="/help.js" defer></script>.
(function () {
  "use strict";

  var HELP_BY_PATH = {
    "/": {
      title: "Áttekintés",
      body: [
        "Ez a főoldal a kereskedési rendszer aktuális állapotát mutatja.",
        "",
        "Felső sáv:",
        " • Egyenleg — a paper trading számla teljes vagyona USD-ben.",
        " • P&L — nyitott pozíciók nem realizált nyeresége/vesztesége.",
        " • Állapot jelző (zöld/sárga/piros) — élő adatfolyam egészsége.",
        "",
        "KPI kártyák: egyenleg, szignálok száma, aktív modellverzió,",
        "átlagos irányvalószínűség és Brier-pont.",
        "",
        "Nyitott pozíciók szakasz: minden aktív LONG/SHORT pozíció külön",
        "sorban; színkód jelzi a nyereséget (zöld) vagy veszteséget (piros).",
        "",
        "Az oldal 2,5 másodpercenként frissül automatikusan."
      ].join("\n")
    },
    "/model.html": {
      title: "Modell diagnosztika",
      body: [
        "A KCA-Mamba modell aktuális állapotát mutatja.",
        "",
        " • Modellverzió: a jelenleg aktív súlyok azonosítója.",
        " • Kalibráció: az előrejelzett valószínűségek illeszkedése.",
        " • Brier-pont: alacsonyabb érték = pontosabb valószínűség.",
        " • Directional accuracy: az irány (fel/le) találati aránya.",
        "",
        "Ha a staging modell jobb a baseline-nál, a training engine",
        "automatikusan aktívra kapcsolja (zero-downtime swap)."
      ].join("\n")
    },
    "/signals.html": {
      title: "Kereskedési szignálok",
      body: [
        "Időrendben listázza az utolsó generált szignálokat.",
        "",
        "Oszlopok:",
        " • Időbélyeg — szignál keletkezésének ideje.",
        " • Irány — BUY / SELL / FLAT.",
        " • Valószínűség — a modell bizonyossága (0..1).",
        " • Kockázat állapot — PASSED / BLOCKED (melyik szabály állította meg).",
        "",
        "A kockázati szabályok (max pozíció, max notional, stb.) a",
        "core/domain/risk.py-ban vannak definiálva."
      ].join("\n")
    },
    "/training.html": {
      title: "Offline tréning",
      body: [
        "Az offline modelltréning állapotát és eredményeit jeleníti meg.",
        "",
        " • Status: idle / running / completed / failed.",
        " • Deployable: az új modell teljesíti-e a minimális kritériumokat.",
        " • Val directional acc: validation halmazon mért irányhelyesség.",
        " • Epoch veszteség görbe: tanulási dinamika vizualizációja.",
        "",
        "A tréning a háttérben fut; a live stream nem szakad meg."
      ].join("\n")
    },
    "/benchmark.html": {
      title: "Szintetikus benchmark",
      body: [
        "Kontrollált, zajjal terhelt szintetikus adatokon méri a modelleket.",
        "",
        " • Zajszint: 0 (tiszta) .. 5 (erős gaussi zaj).",
        " • Modell MSE: alacsonyabb érték = jobb predikció.",
        " • Degradáció: a zajra való érzékenység százalékban."
      ].join("\n")
    },
    "/ltsf_benchmark.html": {
      title: "LTSF összehasonlítás",
      body: [
        "Long-term time series forecasting adatokon (Exchange, ETTh,",
        "ETTm, ECL, Weather, M4) méri az LSTM, Fair Mamba, KCA-Mamba,",
        "ARIMA modelleket.",
        "",
        " • Stride: az ablakok közti lépés (1=nagy átfedés, 32=független).",
        " • Méret: small / medium / large paraméterszám-tartomány.",
        " • KCA-Mamba vs Fair Mamba oszlop: azonos paraméterszám mellett",
        "   melyik modell adott alacsonyabb MSE-t."
      ].join("\n")
    },
    "/kla_kalman.html": {
      title: "Kalman-szűrő diagnosztika",
      body: [
        "A KCA-Mamba Kalman-elemének belső állapotait mutatja.",
        "",
        " • K (Kalman gain) mean: az átlagos súlyozás a megfigyelés",
        "   és a predikció között. Magas = megfigyeléseket követi,",
        "   alacsony = a belső modellre támaszkodik.",
        " • A mátrix: állapotátmenet sajátértékei.",
        " • R (mérési zaj): a bemeneti jelben feltételezett zajszint.",
        " • Residual gate: a reziduál hálózati súlya."
      ].join("\n")
    },
    "/complexity.html": {
      title: "Modell-komplexitás térkép",
      body: [
        "Paraméterszám × teljesítmény szórásdiagram.",
        "",
        "Minden pont egy (modell, méret, dataset) kombinációt reprezentál.",
        "A bal alsó sarok a 'jó': kevés paraméter, alacsony MSE."
      ].join("\n")
    }
  };

  var DEFAULT_HELP = {
    title: "KCA-Mamba dashboard — Súgó",
    body: [
      "Helyzet-érzekeny súgó minden oldalon a jobb alsó '?' gombbal.",
      "",
      "Menü:",
      " • Pénzügyi — Áttekintés, modell, szignálok, tréning.",
      " • Benchmark — szintetikus és valódi datasetek.",
      " • KCA elemzés — Kalman-szűrő belső állapotai, komplexitás.",
      "",
      "Leállítás: a konzolon Ctrl+C."
    ].join("\n")
  };

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function injectStyles() {
    if (document.getElementById("help-overlay-style")) return;
    var css = [
      "#helpBtn{position:fixed;right:20px;bottom:20px;width:44px;height:44px;border-radius:50%;",
      "border:1px solid #1e3a5f;background:#172554;color:#93c5fd;font-size:20px;font-weight:700;",
      "cursor:pointer;z-index:9998;box-shadow:0 4px 16px rgba(0,0,0,.5);transition:transform .15s,background .15s}",
      "#helpBtn:hover{background:#1e3a8a;transform:scale(1.08);color:#dbeafe}",
      "#helpBtn:focus{outline:2px solid #60a5fa;outline-offset:2px}",
      "#helpOverlay{position:fixed;inset:0;background:rgba(2,6,23,.75);display:none;",
      "align-items:center;justify-content:center;z-index:9999;padding:20px;backdrop-filter:blur(4px)}",
      "#helpOverlay.open{display:flex}",
      "#helpBox{background:#0f172a;border:1px solid #334155;border-radius:14px;max-width:640px;",
      "width:100%;max-height:80vh;overflow:auto;padding:24px;color:#e2e8f0;",
      "font-family:ui-sans-serif,system-ui,sans-serif;line-height:1.5}",
      "#helpBox h2{font-size:18px;font-weight:700;color:#60a5fa;margin:0 0 12px}",
      "#helpBox pre{font-family:ui-monospace,monospace;font-size:13px;color:#cbd5e1;",
      "white-space:pre-wrap;margin:0}",
      "#helpClose{float:right;border:0;background:none;color:#94a3b8;font-size:22px;",
      "cursor:pointer;line-height:1;padding:0 4px}",
      "#helpClose:hover{color:#f1f5f9}"
    ].join("");
    var style = document.createElement("style");
    style.id = "help-overlay-style";
    style.textContent = css;
    document.head.appendChild(style);
  }

  function build() {
    injectStyles();

    var btn = document.createElement("button");
    btn.id = "helpBtn";
    btn.type = "button";
    btn.setAttribute("aria-label", "Súgó megnyitása");
    btn.title = "Súgó (F1)";
    btn.textContent = "?";

    var overlay = document.createElement("div");
    overlay.id = "helpOverlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Súgó");

    var box = document.createElement("div");
    box.id = "helpBox";
    overlay.appendChild(box);

    document.body.appendChild(btn);
    document.body.appendChild(overlay);

    function open() {
      var help = HELP_BY_PATH[location.pathname] || DEFAULT_HELP;
      box.innerHTML =
        '<button id="helpClose" type="button" aria-label="Bezárás">×</button>' +
        "<h2>" + esc(help.title) + "</h2>" +
        "<pre>" + esc(help.body) + "</pre>";
      overlay.classList.add("open");
      document.getElementById("helpClose").addEventListener("click", close);
    }

    function close() {
      overlay.classList.remove("open");
    }

    btn.addEventListener("click", open);
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) close();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
      if (e.key === "F1") { e.preventDefault(); open(); }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();

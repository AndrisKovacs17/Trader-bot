// Shared card tooltip system for all dashboard pages.
//
// Usage: <script src="/tooltip.js" defer></script>
// Then on any element:  <button class="tip-btn" data-tip="...">?</button>
//
// The tooltip supports multi-line text via \n in data-tip.
(function () {
  "use strict";

  // Inject CSS once.
  var css = [
    ".tip-btn{width:15px;height:15px;border-radius:50%;border:1px solid #334155;",
    "background:transparent;color:#475569;font-size:9px;font-weight:700;cursor:pointer;",
    "display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;",
    "line-height:1;padding:0;transition:all .15s;font-family:inherit;margin-left:6px;}",
    ".tip-btn:hover{background:#1e293b;color:#93c5fd;border-color:#475569;}",
    "#gTooltip{position:fixed;z-index:9999;max-width:320px;background:#0c1322;",
    "border:1px solid #2d3f5e;border-radius:10px;padding:11px 15px;font-size:12px;",
    "color:#cbd5e1;line-height:1.65;pointer-events:none;white-space:pre-line;",
    "box-shadow:0 8px 36px rgba(0,0,0,.7);display:none;}"
  ].join("");
  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // Tooltip element.
  var tt = document.createElement("div");
  tt.id = "gTooltip";
  document.body.appendChild(tt);

  document.addEventListener("mouseover", function (e) {
    var b = e.target.closest(".tip-btn");
    if (!b) return;
    tt.textContent = b.dataset.tip || "";
    tt.style.display = "block";
  });

  document.addEventListener("mousemove", function (e) {
    if (tt.style.display === "none") return;
    var x = e.clientX + 16, y = e.clientY + 16;
    if (x + 330 > window.innerWidth) x = e.clientX - 334;
    if (y + tt.offsetHeight + 8 > window.innerHeight) y = e.clientY - tt.offsetHeight - 8;
    if (x < 0) x = 4;
    if (y < 0) y = 4;
    tt.style.left = x + "px";
    tt.style.top = y + "px";
  });

  document.addEventListener("mouseout", function (e) {
    var b = e.target.closest(".tip-btn");
    if (!b) return;
    if (!e.relatedTarget || !e.relatedTarget.closest(".tip-btn")) tt.style.display = "none";
  });
})();

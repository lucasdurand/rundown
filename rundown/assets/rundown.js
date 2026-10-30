(function () {
  var root = document.getElementById("__RUNDOWN_ID__");
  if (!root || root.classList.contains("rd-js")) return;
  root.classList.add("rd-js");
  var tbody = root.querySelector("tbody");
  var original = Array.prototype.slice.call(tbody.rows);
  var heads = root.querySelectorAll("th.rd-col");
  var state = { col: -1, dir: 0 };

  function key(row, c) {
    var cell = row.cells[c];
    if (cell.hasAttribute("data-null")) return null;
    var v = cell.getAttribute("data-v");
    if (v !== null) return parseFloat(v);
    return (cell.getAttribute("title") || cell.textContent).toLowerCase();
  }

  Array.prototype.forEach.call(heads, function (th) {
    th.addEventListener("click", function () {
      var c = +th.getAttribute("data-col");
      state = state.col === c ? { col: c, dir: state.dir === 1 ? -1 : state.dir === -1 ? 0 : 1 } : { col: c, dir: 1 };
      Array.prototype.forEach.call(heads, function (h) { h.removeAttribute("aria-sort"); });
      var rows = original.slice();
      if (state.dir) {
        th.setAttribute("aria-sort", state.dir > 0 ? "ascending" : "descending");
        rows.sort(function (a, b) {
          var x = key(a, c), y = key(b, c);
          if (x === null) return y === null ? 0 : 1;
          if (y === null) return -1;
          return (x < y ? -1 : x > y ? 1 : 0) * state.dir;
        });
      }
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  });

  var filter = root.querySelector(".rd-filter");
  filter.addEventListener("input", function () {
    var q = filter.value.trim().toLowerCase();
    original.forEach(function (r) {
      r.style.display = !q || r.textContent.toLowerCase().indexOf(q) !== -1 ? "" : "none";
    });
  });
})();

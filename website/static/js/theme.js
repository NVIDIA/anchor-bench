(function () {
  var KEY = "ab-theme";
  var VALID = ["light", "dark", "novel"];

  function current() {
    var t = document.documentElement.getAttribute("data-theme");
    return VALID.indexOf(t) === -1 ? "light" : t;
  }

  function apply(theme) {
    if (VALID.indexOf(theme) === -1) theme = "light";
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch (e) {}
    var btns = document.querySelectorAll("[data-theme-btn]");
    for (var i = 0; i < btns.length; i++) {
      var b = btns[i];
      b.setAttribute(
        "aria-pressed",
        b.getAttribute("data-theme-btn") === theme ? "true" : "false"
      );
    }
  }

  function init() {
    var q = null;
    try {
      q = new URLSearchParams(window.location.search).get("theme");
    } catch (e) {}
    apply(VALID.indexOf(q) !== -1 ? q : current());
    var btns = document.querySelectorAll("[data-theme-btn]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener("click", function () {
        apply(this.getAttribute("data-theme-btn"));
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

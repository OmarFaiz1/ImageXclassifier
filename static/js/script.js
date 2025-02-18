function showLoading(mode) {
  if (mode === "single" || mode === "bulk") {
    document.getElementById("loadingMessage").style.display = "block";
  }
}

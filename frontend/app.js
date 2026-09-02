const form = document.getElementById("search-form");
const input = document.getElementById("query");
const button = document.getElementById("submit");
const result = document.getElementById("result");

function show(label, text, isError = false) {
  result.hidden = false;
  result.classList.toggle("error", isError);
  result.innerHTML = "";

  const labelEl = document.createElement("p");
  labelEl.className = "label";
  labelEl.textContent = label;

  const textEl = document.createElement("p");
  textEl.className = "echo";
  textEl.textContent = text;

  result.append(labelEl, textEl);
}

function showResults(results) {
  result.hidden = false;
  result.classList.remove("error");
  result.innerHTML = "";

  const labelEl = document.createElement("p");
  labelEl.className = "label";
  labelEl.textContent = results.length ? "Closest matches" : "No matches";
  result.append(labelEl);

  const list = document.createElement("ol");
  list.className = "hits";

  for (const hit of results) {
    const item = document.createElement("li");
    item.className = "hit";

    const head = document.createElement("div");
    head.className = "hit-head";

    const title = document.createElement("h2");
    title.className = "hit-title";
    title.textContent = hit.year ? `${hit.title} (${hit.year})` : hit.title;

    const match = document.createElement("span");
    match.className = "hit-match";
    match.textContent = `${hit.match}% match`;

    head.append(title, match);

    const overview = document.createElement("p");
    overview.className = "hit-overview";
    overview.textContent = hit.overview;

    item.append(head, overview);
    list.append(item);
  }

  result.append(list);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = input.value.trim();
  if (!query) {
    input.focus();
    return;
  }

  button.disabled = true;
  button.textContent = "Searching…";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await response.json();

    if (!response.ok) {
      show("Error", data.error || "Something went wrong.", true);
    } else {
      showResults(data.results || []);
    }
  } catch (err) {
    show("Error", "Could not reach the server.", true);
  } finally {
    button.disabled = false;
    button.textContent = "Search";
  }
});

// ⌘/Ctrl + Enter submits from inside the textarea.
input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    form.requestSubmit();
  }
});

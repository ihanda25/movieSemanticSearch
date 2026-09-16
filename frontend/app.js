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

function showResults(results, searchId) {
  result.hidden = false;
  result.classList.remove("error");
  result.innerHTML = "";

  const labelEl = document.createElement("p");
  labelEl.className = "label";
  labelEl.textContent = results.length ? "Closest matches" : "No matches";
  result.append(labelEl);

  const status = document.createElement("p");
  status.className = "feedback-status";
  status.setAttribute("role", "status");
  status.textContent = "Found your movie? Your choice is saved locally to improve future search.";
  let saving = false;
  const selections = [];
  async function save(outcome, movieId, control) {
    if (saving) return;
    saving = true;
    const previousLabel = control.textContent;
    control.textContent = "Saving…";
    control.disabled = true;
    status.textContent = "Saving feedback…";
    try {
      const response = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ search_id: searchId, outcome, movie_id: movieId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not save feedback.");
      for (const selection of selections) {
        selection.textContent = selection.dataset.label;
        selection.setAttribute("aria-pressed", "false");
      }
      control.textContent = "Saved ✓";
      control.setAttribute("aria-pressed", "true");
      status.textContent = outcome === "none"
        ? "Saved: none of these. You can identify the correct title below."
        : "Saved your selection. Choose again to correct it.";
    } catch (error) {
      control.textContent = previousLabel;
      status.textContent = `Feedback not saved: ${error.message}`;
      control.after(status);
    }
    finally { saving = false; control.disabled = false; }
  }
  function choice(label, action) {
    const control = document.createElement("button");
    control.type = "button";
    control.className = "feedback-choice";
    control.textContent = label;
    control.addEventListener("click", () => action(control));
    return control;
  }

  function selection(label, outcome, movieId) {
    const control = choice(label, button => save(outcome, movieId, button));
    control.dataset.label = label;
    control.setAttribute("aria-pressed", "false");
    selections.push(control);
    return control;
  }

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

    item.append(head, overview, selection("This is it", "selected", hit.id));
    list.append(item);
  }

  const feedback = document.createElement("div");
  feedback.className = "feedback";
  const titleInput = document.createElement("input");
  titleInput.placeholder = "Know the correct movie? Enter its title";
  titleInput.setAttribute("aria-label", "Correct movie title");
  const alternatives = document.createElement("div");
  async function lookup() {
    const q = titleInput.value.trim();
    if (!q) { titleInput.focus(); return; }
    alternatives.textContent = "Looking up titles…";
    try {
      const response = await fetch(`/api/movies?q=${encodeURIComponent(q)}`);
      if (!response.ok) throw new Error("Title lookup failed.");
      const data = await response.json();
      alternatives.textContent = data.results.length ? "Choose the correct movie (up to 30 matches):" : "No catalog titles found. Try a shorter title.";
      for (const movie of data.results) {
        alternatives.append(selection(`${movie.title} (${movie.year || "year unknown"})`, "other", movie.id));
      }
    } catch (error) { alternatives.textContent = error.message; }
  }
  titleInput.addEventListener("keydown", event => {
    if (event.key === "Enter") { event.preventDefault(); lookup(); }
  });
  feedback.append(selection("None of these", "none"), status,
    titleInput, choice("Find title", lookup), alternatives);
  result.append(list, feedback);
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
      showResults(data.results || [], data.search_id);
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

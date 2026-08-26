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
      show("Sent to the server", data.received);
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

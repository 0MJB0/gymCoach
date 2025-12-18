function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function scrollBottom() {
    const box = document.getElementById("chatBox");
    box.scrollTop = box.scrollHeight;
}

function addMessage(role, text) {
    const box = document.getElementById("chatBox");
    const div = document.createElement("div");
    div.classList.add("chat-message", role);

    div.innerHTML = `
        <div class="bubble">
            <strong>${role === "user" ? "User" : "Assistant"}:</strong> ${text}
        </div>
    `;

    box.appendChild(div);
    scrollBottom();
}

function addTyping() {
    const box = document.getElementById("chatBox");
    const div = document.createElement("div");
    div.id = "typingBubble";
    div.classList.add("chat-message", "assistant");

    div.innerHTML = `
        <div class="bubble typing">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
    `;

    box.appendChild(div);
    scrollBottom();
}

function removeTyping() {
    const t = document.getElementById("typingBubble");
    if (t) t.remove();
}

function sendMessage() {
    const fileInput = document.getElementById("logFileInput");
    const input = document.getElementById("chatInput");
    const text = input.value.trim();
    const hasFile = fileInput.files.length > 0;

    if (!text && !hasFile) {
        return;
    }

    if (text) {
        addMessage("user", text);
    }

    input.value = "";
    addTyping();

    const formData = new FormData();
    if (text) formData.append("message", text);
    if (hasFile) formData.append("logfile", fileInput.files[0]);

    fetch("/send", { method: "POST", body: formData })
        .then(r => r.json())
        .then(data => {
            removeTyping();
            addMessage("assistant", data.response);
            fileInput.value = "";
            refreshWorkouts();
        })
        .catch(err => {
            removeTyping();
            addMessage("assistant", "Server error, try again.");
            console.error("FETCH ERROR:", err);
        });
}

// Auto-send when a CSV is selected
const uploadInput = document.getElementById("logFileInput");
if (uploadInput) {
    uploadInput.addEventListener("change", () => {
        if (uploadInput.files.length > 0) {
            sendMessage();
        }
    });
}

// ----------------------------
// WORKOUT HISTORY WITH CHECKBOXES
// ----------------------------
let workoutsCache = [];

function showWorkout(workout, label) {
    const report = workout.report || workout.summary || "";
    let htmlReport = report;

    const title = label ? `🏋️ Workout Analysis - ${label}` : "🏋️ Workout Analysis";
    if (htmlReport.includes("🏋️ Workout Analysis")) {
        htmlReport = htmlReport.replace("🏋️ Workout Analysis", title);
    } else {
        htmlReport = `${title}\n${htmlReport}`;
    }

    let html = htmlReport.replace(/\n/g, "<br>");

    if (workout.acc_graph) {
        html += `<br><img src="${workout.acc_graph}" style="max-width:100%">`;
    }

    if (workout.tempo_graph) {
        html += `<br><img src="${workout.tempo_graph}" style="max-width:100%">`;
    }

    addMessage("assistant", html);
}

function refreshWorkouts() {
    fetch("/workouts")
        .then(res => res.json())
        .then(data => {
            workoutsCache = data;
            const list = document.getElementById("workoutList");
            list.innerHTML = "";

            data.slice().reverse().forEach((w, i) => {
                const actualIndex = data.length - 1 - i; // index in original array
                const labelNum = data.length - actualIndex;

                const row = document.createElement("div");
                row.className = "workout-row";

                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.className = "workout-check";
                checkbox.value = actualIndex;
                checkbox.id = `workout-${labelNum}`;

                const btn = document.createElement("button");
                btn.innerText = `Workout ${labelNum}`;
                btn.onclick = () => showWorkout(w, `Workout ${labelNum}`);

                row.appendChild(checkbox);
                row.appendChild(btn);
                list.appendChild(row);
            });
        })
        .catch(() => {
            // history is optional; fail quietly
        });
}

refreshWorkouts();

function compareWorkouts() {
    const selected = Array.from(document.querySelectorAll(".workout-check:checked"))
        .map(el => parseInt(el.value, 10))
        .filter(n => !Number.isNaN(n));

    if (selected.length < 2) {
        addMessage("assistant", "Select at least two workouts to compare.");
        return;
    }

    const entries = selected
        .map(idx => ({ idx, data: workoutsCache[idx] }))
        .filter(({ data }) => !!data);

    const lines = ["Comparison:"];
    let bestForm = null;

    entries.forEach(({ idx, data }) => {
        const number = workoutsCache.length - idx;
        const a = data.analysis || {};
        const reps = a.rep_count ?? "-";
        const sets = a.sets ?? "-";
        const form = a.form_score ?? "-";
        const issue = a.issue ?? "n/a";
        const rec = a.recommendation ?? "n/a";

        lines.push(
            `Workout ${number}: reps ${reps}, sets ${sets}, form ${form}%, issue: ${issue}, cue: ${rec}`
        );

        if (typeof a.form_score === "number") {
            if (!bestForm || a.form_score > bestForm.form_score) {
                bestForm = { number, form_score: a.form_score };
            }
        }
    });

    if (bestForm) {
        lines.push(
            `Best form: Workout ${bestForm.number} (${bestForm.form_score}%)`
        );
    }

    addMessage("assistant", lines.join("\n"));
}

/* The trial page.
 *
 * Two jobs: record how long each answer took, and make the ranking task
 * usable. Response time is one of the study's primary measures -- the whole
 * comparison is per unit of participant time -- so it is measured here, in the
 * page, from when the trial is shown to when it is answered. Measuring it
 * server-side would fold in network latency and page rendering, which are not
 * the participant's thinking time.
 *
 * Without this file the trial still works: the pairwise buttons submit their
 * own value, and the ranking has real number inputs. Only the drag-to-reorder
 * and the timing are lost.
 */
(function () {
    "use strict";

    var form = document.getElementById("trial-form");
    if (!form) {
        return;
    }

    var shownAt = performance.now();
    var answerField = document.getElementById("answer");
    var elapsedField = document.getElementById("elapsed_ms");
    var list = document.getElementById("rank-list");

    function stampElapsed() {
        elapsedField.value = Math.round(performance.now() - shownAt);
    }

    // --- Pairwise ---------------------------------------------------------
    form.addEventListener("click", function (event) {
        var choice = event.target.closest ? event.target.closest(".movie-choice") : null;
        if (!choice) {
            return;
        }
        stampElapsed();
        answerField.value = choice.getAttribute("data-answer");
    });

    // --- Ranking ----------------------------------------------------------
    if (!list) {
        form.addEventListener("submit", stampElapsed);
        return;
    }

    function items() {
        return Array.prototype.slice.call(list.querySelectorAll(".rank-item"));
    }

    function renumber() {
        items().forEach(function (item, index) {
            item.querySelector(".rank-position").textContent = index + 1;
            item.querySelector("input").value = index + 1;
        });
    }

    var dragging = null;

    list.addEventListener("dragstart", function (event) {
        dragging = event.target.closest(".rank-item");
        if (!dragging) {
            return;
        }
        dragging.classList.add("dragging");
        event.dataTransfer.effectAllowed = "move";
        // Firefox will not start a drag without data set on the transfer.
        event.dataTransfer.setData("text/plain", dragging.dataset.title || "");
    });

    list.addEventListener("dragend", function () {
        if (dragging) {
            dragging.classList.remove("dragging");
        }
        items().forEach(function (item) { item.classList.remove("drop-target"); });
        dragging = null;
        renumber();
    });

    list.addEventListener("dragover", function (event) {
        event.preventDefault();
        var over = event.target.closest ? event.target.closest(".rank-item") : null;
        if (!over || !dragging || over === dragging) {
            return;
        }
        items().forEach(function (item) { item.classList.remove("drop-target"); });
        over.classList.add("drop-target");

        var box = over.getBoundingClientRect();
        var below = event.clientY > box.top + box.height / 2;
        list.insertBefore(dragging, below ? over.nextSibling : over);
    });

    // Typing a position moves the row, so the two controls never disagree.
    list.addEventListener("change", function (event) {
        var input = event.target;
        if (input.type !== "number") {
            return;
        }
        var row = input.closest(".rank-item");
        var all = items();
        var wanted = Math.min(Math.max(parseInt(input.value, 10) || 1, 1), all.length);

        row.remove();
        var remaining = items();
        var before = remaining[wanted - 1];
        if (before) {
            list.insertBefore(row, before);
        } else {
            list.appendChild(row);
        }
        renumber();
    });

    form.addEventListener("submit", function () {
        stampElapsed();
        // The answer is the ordering, best first.
        answerField.value = items().map(function (item) {
            return item.dataset.title;
        }).join(" > ");
    });
}());

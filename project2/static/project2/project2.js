/* Project 2 - keep the page in place while the controls change.
 *
 * Without this file the page still works: every control belongs to the
 * #p2-controls form and Update performs an ordinary GET. This only makes the
 * interaction better, so nothing here is load-bearing.
 *
 * Each control declares what it affects via data-affects. Changing the feature
 * dropdown re-renders the effect plots alone rather than the tree and the
 * trade-off curve as well, which is both faster and far less disorienting than
 * being thrown back to the top of the page.
 */
(function () {
    "use strict";

    var form = document.getElementById("p2-controls");
    if (!form || !window.fetch) {
        return; // fall back to plain form submission
    }

    // Marks the page as enhanced, which hides the no-JS Update button.
    document.documentElement.classList.add("js-on");

    var REGIONS = ["model", "counterfactuals", "effects"];
    var shell = document.querySelector(".project-shell");
    var inFlight = null;

    function section(name) {
        return document.getElementById("region-" + name);
    }

    function controls() {
        return Array.prototype.slice.call(
            document.querySelectorAll("[form='p2-controls'][name]")
        );
    }

    function currentQuery() {
        var params = new URLSearchParams();
        controls().forEach(function (el) {
            if (el.name) {
                params.set(el.name, el.value);
            }
        });
        return params;
    }

    function busy(names, state) {
        names.forEach(function (name) {
            var el = section(name);
            if (el) {
                el.classList.toggle("is-busy", state);
                el.setAttribute("aria-busy", state ? "true" : "false");
            }
        });
    }

    function refresh(names) {
        var params = currentQuery();

        // Keep the address bar in step, so the view stays linkable and the back
        // button returns to the previous state rather than the previous page.
        history.replaceState(null, "", location.pathname + "?" + params.toString());

        if (inFlight) {
            inFlight.abort();
        }
        var controller = new AbortController();
        inFlight = controller;

        busy(names, true);

        var requests = names.map(function (name) {
            var query = new URLSearchParams(params);
            query.set("partial", name);
            return fetch(location.pathname + "?" + query.toString(), {
                signal: controller.signal,
                headers: { "X-Requested-With": "fetch" }
            })
                .then(function (response) {
                    if (!response.ok) {
                        throw new Error(response.status + " on " + name);
                    }
                    return response.text();
                })
                .then(function (html) {
                    var el = section(name);
                    if (el) {
                        el.innerHTML = html;
                    }
                });
        });

        Promise.all(requests)
            .catch(function (error) {
                if (error.name === "AbortError") {
                    return; // superseded by a newer change
                }
                // Something went wrong server-side; fall back to a real
                // navigation so the user still sees a correct page.
                console.error("project2: partial refresh failed", error);
                form.submit();
            })
            .then(function () {
                busy(names, false);
                inFlight = null;
            });
    }

    function affected(el) {
        var value = el.getAttribute("data-affects");
        if (!value) {
            return [];
        }
        return value === "all" ? REGIONS.slice() : [value];
    }

    // Controls are replaced wholesale when a region refreshes, so listen on a
    // stable ancestor rather than binding to elements that will not survive.
    shell.addEventListener("change", function (event) {
        var el = event.target;
        if (!el.matches || !el.matches("[form='p2-controls'][data-affects]")) {
            return;
        }
        var names = affected(el);
        if (names.length) {
            refresh(names);
        }
    });

    // Live readout while the slider is dragged; the fetch waits for release.
    shell.addEventListener("input", function (event) {
        var el = event.target;
        if (el.id !== "lam") {
            return;
        }
        var readout = document.querySelector("[data-lambda-readout]");
        if (readout) {
            readout.textContent = el.value;
        }
    });

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        refresh(REGIONS.slice());
    });
}());

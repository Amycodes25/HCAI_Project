/* Project 1 - submit the forms without throwing the page away.
 *
 * Every action here is a POST that changes server-side session state (the
 * uploaded dataset), so unlike Project 2 this cannot be a GET against a
 * fragment. The page is still rendered by one view, so the enhancement posts
 * the form with fetch, takes the .project-shell out of the response, and swaps
 * it in. The result is the same HTML the server would have sent on a full
 * navigation -- the browser simply never discards the page.
 *
 * Without this file every form still works exactly as before: these are
 * ordinary POST forms and the view does not care how it was reached.
 */
(function () {
    "use strict";

    var shell = document.querySelector(".dashboard-main") ||
                document.querySelector(".project-shell");
    if (!shell || !window.fetch || !window.FormData || !window.DOMParser) {
        return;
    }

    document.documentElement.classList.add("js-on");

    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var pending = false;

    function busy(state) {
        shell.classList.toggle("is-busy", state);
        shell.setAttribute("aria-busy", state ? "true" : "false");
    }

    function scrollToElement(element) {
        element.scrollIntoView({
            behavior: reduceMotion ? "auto" : "smooth",
            block: "start"
        });
    }

    function reveal(id, previousScroll) {
        var failed = shell.querySelector(".error-message");
        if (failed) {
            // The action did not do what was asked, so show why rather than
            // jumping to a section that has not changed.
            scrollToElement(failed);
            return;
        }

        var target = id && document.getElementById(id);
        if (target) {
            scrollToElement(target);
            return;
        }

        // Restoring a scroll offset has to wait for the swapped-in content to
        // be laid out, or the browser clamps it against a shorter document.
        requestAnimationFrame(function () {
            window.scrollTo(0, previousScroll);
        });
    }

    function swap(html) {
        var parsed = new DOMParser().parseFromString(html, "text/html");
        var fresh = parsed.querySelector(".dashboard-main") ||
                    parsed.querySelector(".project-shell");
        if (!fresh) {
            return false;
        }
        shell.innerHTML = fresh.innerHTML;

        // The sidebar tracks progress through the pipeline, so it has to come
        // across too -- it sits outside the main column and would otherwise
        // keep reporting the state before the action.
        var sidebar = document.querySelector(".dashboard-sidebar");
        var freshSidebar = parsed.querySelector(".dashboard-sidebar");
        if (sidebar && freshSidebar) {
            sidebar.innerHTML = freshSidebar.innerHTML;
        }
        return true;
    }

    shell.addEventListener("submit", function (event) {
        var form = event.target;
        if (!form.matches || !form.matches("form[method='POST' i]")) {
            return;
        }
        if (pending) {
            event.preventDefault();
            return;
        }

        event.preventDefault();
        pending = true;

        var scrollBefore = window.scrollY;
        var revealId = form.getAttribute("data-reveal");

        busy(true);

        fetch(window.location.pathname, {
            method: "POST",
            body: new FormData(form),
            headers: { "X-Requested-With": "fetch" },
            credentials: "same-origin"
        })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error("status " + response.status);
                }
                return response.text();
            })
            .then(function (html) {
                if (!swap(html)) {
                    throw new Error("no page container in the response");
                }
                reveal(revealId, scrollBefore);
            })
            .catch(function (error) {
                // Never leave the user looking at stale results: fall back to a
                // real submission, which is what would have happened anyway.
                console.error("project1: falling back to full submit", error);
                form.submit();
            })
            .then(function () {
                busy(false);
                pending = false;
            });
    });
}());

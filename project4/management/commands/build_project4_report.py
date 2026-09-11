"""Generate the Project 4 report PDF.

The brief asks for a PDF describing the implemented method (Tasks 1 and 2) and
the design of the user study (Task 3), downloadable from the project interface.

It is generated rather than written by hand so that it cannot drift from the
code: the feature list, the number of films and the prior strength are read from
the implementation at build time, not retyped.

    python manage.py build_project4_report
"""

from pathlib import Path

import matplotlib
from django.core.management.base import BaseCommand
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.lib import colors

from project4.ml import features, pilot, preference

OUTPUT = Path(__file__).resolve().parents[2] / "static" / "project4" / "report.pdf"

INK = colors.HexColor("#002429")
MUTED = colors.HexColor("#58686e")
LINE = colors.HexColor("#cfd9e6")
ACCENT = colors.HexColor("#135e78")


BODY_FONT = "DejaVuSans"


def register_fonts():
    """Use DejaVu Sans, because the report contains mathematics.

    Helvetica has no glyph for the product, sum, alpha or true minus signs, and
    ReportLab draws nothing at all where a glyph is missing rather than warning.
    The Plackett-Luce product and the MAP objective were therefore rendering as
    lines with holes in them -- which is unfortunate in a report whose Task 2
    deliverable is a formulation.

    DejaVu ships inside matplotlib, which is already a runtime requirement, so
    this costs no new dependency and no font file committed to the repository.
    """
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    faces = {
        BODY_FONT: "DejaVuSans.ttf",
        f"{BODY_FONT}-Bold": "DejaVuSans-Bold.ttf",
        f"{BODY_FONT}-Oblique": "DejaVuSans-Oblique.ttf",
        f"{BODY_FONT}-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    }
    for name, filename in faces.items():
        pdfmetrics.registerFont(TTFont(name, str(ttf / filename)))

    # Without the mapping, <b> and <i> inside a Paragraph silently fall back to
    # the regular face instead of the bold or italic one.
    addMapping(BODY_FONT, 0, 0, BODY_FONT)
    addMapping(BODY_FONT, 1, 0, f"{BODY_FONT}-Bold")
    addMapping(BODY_FONT, 0, 1, f"{BODY_FONT}-Oblique")
    addMapping(BODY_FONT, 1, 1, f"{BODY_FONT}-BoldOblique")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=21,
                                leading=25, textColor=INK, spaceAfter=4,
                                fontName=f"{BODY_FONT}-Bold"),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=10.5,
                                   leading=14, textColor=MUTED, spaceAfter=18,
                                   fontName=BODY_FONT),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14,
                             leading=18, textColor=INK, spaceBefore=16,
                             spaceAfter=6, fontName=f"{BODY_FONT}-Bold"),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11,
                             leading=15, textColor=ACCENT, spaceBefore=12,
                             spaceAfter=4, fontName=f"{BODY_FONT}-Bold"),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5,
                               leading=14.5, textColor=INK, alignment=TA_JUSTIFY,
                               spaceAfter=7, fontName=BODY_FONT),
        "formula": ParagraphStyle("f", parent=base["BodyText"], fontSize=10,
                                  leading=15, textColor=INK, alignment=1,
                                  spaceBefore=6, spaceAfter=10,
                                  fontName=f"{BODY_FONT}-Oblique"),
        "small": ParagraphStyle("s", parent=base["BodyText"], fontSize=8.5,
                                leading=12, textColor=MUTED, spaceAfter=6,
                                fontName=BODY_FONT),
        # Table cells. These have to be Paragraph styles rather than TableStyle
        # commands, because only a Paragraph wraps inside a fixed column.
        "cell_label": ParagraphStyle("cl", parent=base["BodyText"], fontSize=8.5,
                                     leading=12, textColor=ACCENT, spaceAfter=0,
                                     fontName=f"{BODY_FONT}-Bold"),
        "cell": ParagraphStyle("cd", parent=base["BodyText"], fontSize=8.5,
                               leading=12, textColor=INK, spaceAfter=0,
                               fontName=BODY_FONT),
    }


def bullets(items, style):
    return ListFlowable(
        [ListItem(Paragraph(text, style), leftIndent=12) for text in items],
        bulletType="bullet", bulletFontSize=6, leftIndent=14, spaceAfter=8,
    )


def table(s, rows):
    """A two-column label/prose table whose prose actually wraps.

    Passing a bare string as a table cell makes ReportLab lay it out as a
    single unbroken line: it does not wrap, it does not shrink, it simply runs
    past the column and off the page. Every row of the design table was being
    cut off mid-sentence. Wrapping each cell in a Paragraph gives the text a
    flowable that knows the column width, which is the whole fix.
    """
    cells = [
        [Paragraph(label, s["cell_label"]), Paragraph(text, s["cell"])]
        for label, text in rows
    ]
    t = Table(cells, colWidths=[4.2 * cm, 11.3 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
    ]))
    return t


class Command(BaseCommand):
    help = "Write the Project 4 report PDF from the implementation"

    def handle(self, *args, **options):
        register_fonts()
        s = styles()
        names = features.feature_names()
        genres = features.genre_summary()
        others = names[features.N_GENRES:]

        story = [
            Paragraph("Preference Elicitation for a Movie Recommender", s["title"]),
            Paragraph(
                "Human-Centric Artificial Intelligence &nbsp;&middot;&nbsp; Group 29 "
                "&nbsp;&middot;&nbsp; Project 4",
                s["subtitle"]),

            Paragraph("Introduction", s["h1"]),
            Paragraph(
                "This report covers the method behind a movie recommender that adapts to a new "
                "user, and the design of a user study comparing two ways of eliciting that "
                "user's preferences. The utility of a film to a user is taken to be linear, "
                "U(x) = w<sup>T</sup>x, and the purpose of elicitation is to estimate w from a "
                "small number of interactions. Section 1 sets out the feature representation, "
                "section 2 the preference model and its estimator, and section 3 the study. "
                "The study is designed, not conducted.", s["body"]),
            Paragraph(
                f"Everything reported here is produced by the implementation in "
                f"<b>project4/ml/</b>, over the {features.movie_count()} films of the IMDB 5000 "
                f"Movie Dataset that carry every field the representation needs.", s["body"]),

            Paragraph("1 &nbsp; Feature representation", s["h1"]),
            Paragraph("1.1 &nbsp; The constraint that decides the design", s["h2"]),
            Paragraph(
                "Two sentences of the specification fix the problem, and they pull against each "
                "other. Because U is linear, whatever the representation does not carry cannot "
                "be learned at all: a taste for a particular director exists only if some "
                "component of x expresses it, which argues for more features. Because w must be "
                "estimated from roughly ten interactions, every extra component is another "
                "number to identify from the same evidence, which argues for fewer. A "
                "representation with one column per director would have thousands of components "
                "and none of them would be estimable.", s["body"]),
            Paragraph(
                f"We therefore kept the representation small, at <b>{features.n_features()} "
                f"components</b>, each chosen because it separates films in a way a person would "
                f"recognise as taste.", s["body"]),

            Paragraph("1.2 &nbsp; What is included", s["h2"]),
            table(s, [
                ["Genre", f"Multi-hot over the {features.N_GENRES} most common genres: "
                          + ", ".join(genres)
                          + ". Multi-label by nature, since a film can be both a comedy and a "
                            "romance, so multi-hot rather than one-hot. The remaining genres in "
                            "the dataset appear on too few films for their weights to be "
                            "estimated in a short session."],
                ["Era", "This is the release year, scaled. Most people have some preference about "
                        "era, and a single scaled number captures the steady part of that for "
                        "the cost of one component. One-hot decades would have cost ten."],
                ["Runtime", "This is the duration, scaled. A ninety-minute film and a three-hour "
                            "film are quite different propositions on a weeknight, and people "
                            "do take that into account."],
                ["Critical standing", "This is the IMDB score, scaled. It separates viewers who "
                                      "follow critical acclaim from those who pay it no "
                                      "attention."],
                ["Popularity", "This is the number of votes, logged and then scaled. It captures "
                               "the difference between mainstream and obscure taste. We take the "
                               "logarithm first because vote counts span five orders of "
                               "magnitude, and on the raw scale a handful of blockbusters would "
                               "dominate the inner product."],
                ["Production scale", "This is the budget, logged and then scaled. It is a "
                                     "different axis from popularity, because expensive films "
                                     "can flop and cheap ones can be seen by everyone."],
                ["Director prominence", "This is the director's follower count, logged and then "
                                        "scaled. It stands in cheaply for a taste in particular "
                                        "directors, without spending a column on each one."],
                ["Family-friendly", "This is a single indicator that is set when the content "
                                    "rating is G or PG. One-hot encoding all twelve ratings "
                                    "would have spent twelve components on a distinction that "
                                    "mostly matters at this one boundary."],
            ]),

            Paragraph("1.3 &nbsp; What is excluded", s["h2"]),
            Paragraph(
                "Director and cast identity, plot keywords and country of production are all "
                "high-cardinality, and the signal in them worth keeping is already carried by "
                "director prominence and popularity. Gross earnings are excluded because they "
                "are largely determined by budget and popularity, both of which are present; a "
                "third correlated column adds collinearity, and collinear features are precisely "
                "what makes w hard to identify from few observations.", s["body"]),

            Paragraph("1.4 &nbsp; Extraction and scaling", s["h2"]),
            Paragraph(
                "Rows missing any required field are dropped and duplicates on title and year "
                "removed. Every continuous column is then standardised. This matters because "
                "utility is an inner product: a column's scale sets the scale of its weight, so "
                "without standardisation the budget column, of order 10<sup>8</sup>, and the "
                "genre columns, which are zero or one, could not share a sensible prior and the "
                "regularisation would penalise them wildly unevenly.", s["body"]),

            Paragraph("2 &nbsp; Preference model", s["h1"]),
            Paragraph("2.1 &nbsp; From a comparison to a ranking", s["h2"]),
            Paragraph(
                "Preferences are assumed to follow a Bradley&ndash;Terry model, which gives the "
                "probability that one film is preferred to another from their utilities:", s["body"]),
            Paragraph("P(i &gt; j) = exp(U<sub>i</sub>) / [ exp(U<sub>i</sub>) + exp(U<sub>j</sub>) ]",
                      s["formula"]),
            Paragraph(
                "One of the two interfaces asks a participant to rank ten films, so the model has "
                "to describe a whole ordering rather than a single comparison. The extension used "
                "is the <b>Plackett&ndash;Luce model</b>, which treats a ranking as produced "
                "sequentially: the participant picks their favourite from the set, then their "
                "favourite of what remains, and so on, each pick following the Luce choice rule.",
                s["body"]),
            Paragraph(
                "P(i<sub>1</sub> &gt; i<sub>2</sub> &gt; &hellip; &gt; i<sub>n</sub>) = "
                "&Pi;<sub>k</sub> exp(U<sub>i<sub>k</sub></sub>) / "
                "&Sigma;<sub>j&ge;k</sub> exp(U<sub>i<sub>j</sub></sub>)", s["formula"]),

            Paragraph("2.2 &nbsp; Why this extension", s["h2"]),
            Paragraph(
                "First, it genuinely contains the model we started from. At n = 2 the product has "
                "a single factor, exp(U<sub>1</sub>) / [exp(U<sub>1</sub>) + exp(U<sub>2</sub>)], "
                "which is exactly Bradley&ndash;Terry. Both interfaces are therefore described by "
                "one model with one w, and their results are directly comparable &mdash; which is "
                "the entire point of the study. An extension that did not reduce would be a "
                "second model in disguise, and the comparison would be confounded by the choice "
                "of model rather than the choice of interface.", s["body"]),
            Paragraph(
                "Second, the generative story it encodes is a plausible account of what a person "
                "actually does when ranking: choose the best, set it aside, choose the best of "
                "the rest.", s["body"]),

            Paragraph("2.3 &nbsp; The alternative that was rejected", s["h2"]),
            Paragraph(
                "A ranking of ten implies 45 pairwise comparisons, and these could be fed to "
                "Bradley&ndash;Terry as if they were separate observations. This needs no new "
                "model and is simpler. It is rejected because those 45 comparisons are not "
                "independent: they arise from one ordering produced by one person in one act. "
                "Treating them as independent multiplies the same evidence many times over, so "
                "the likelihood is misspecified and the apparent precision of w is badly "
                "overstated. In a study whose purpose is to compare how much each interface "
                "reveals about w, an estimator that inflates its own confidence for one of the "
                "two designs would decide the outcome before any data was collected.", s["body"]),

            Paragraph("2.4 &nbsp; Estimating w", s["h2"]),
            Paragraph(
                "The log-likelihood of a set of observed rankings is concave in w. It is "
                "maximised under a Gaussian prior:", s["body"]),
            Paragraph(
                "w&#770; = argmax &nbsp; &Sigma;<sub>r</sub> log P(ranking<sub>r</sub> | w) "
                f"&nbsp;&minus;&nbsp; (&alpha;/2) ||w||<sup>2</sup>, &nbsp; &alpha; = "
                f"{preference.DEFAULT_ALPHA}", s["formula"]),
            Paragraph(
                f"The prior does real work here. With roughly ten interactions and "
                f"{features.n_features()} features the unpenalised maximum is not unique: any "
                f"direction the shown films do not distinguish is unconstrained, and the "
                f"optimiser will run off along it. The penalty holds those directions at zero, "
                f"which is the honest statement that the data said nothing about them. "
                f"Optimisation uses L-BFGS with the analytic gradient.", s["body"]),
            Paragraph(
                "Two properties were checked against the implementation. At n = 2 the "
                "Plackett&ndash;Luce probability equals the Bradley&ndash;Terry probability to "
                "10<sup>&minus;12</sup>. The analytic gradient agrees with central differences to "
                "10<sup>&minus;8</sup>.", s["small"]),

            Paragraph("3 &nbsp; User study", s["h1"]),
            Paragraph("3.1 &nbsp; Research question and hypothesis", s["h2"]),
            Paragraph(
                "The study compares two elicitation interfaces. In Design 1 the participant is "
                "shown two films and chooses the one they would rather watch. In Design 2 they "
                "are shown ten films and rank them.", s["body"]),
            Paragraph(
                "The two are not equally expensive, and this is the crux of the design. A "
                "pairwise choice yields at most one bit and takes seconds. Ranking ten yields up "
                "to log<sub>2</sub>(10!) &asymp; 21.8 bits but takes considerably longer and asks "
                "more of the participant. So the question &lsquo;which elicits w better&rsquo; is "
                "under-specified: <b>better per what?</b> Per interaction, ranking wins almost by "
                "construction and the study would be pointless. Per minute of participant time it "
                "is a genuine question. Fixing the budget by which the comparison is normalised "
                "is the single most consequential decision in this design.", s["body"]),
            Paragraph(
                "<b>H1.</b> For a fixed amount of participant time, ranking ten films yields a "
                "more accurate estimate of w than repeated pairwise choices.<br/>"
                "<b>H0.</b> There is no difference in accuracy per unit of participant time.",
                s["body"]),

            Paragraph("3.2 &nbsp; Design", s["h2"]),
            table(s, [
                ["Type", "Experimental. The interface is manipulated directly and assignment is "
                         "controlled, which permits a causal claim; an observational comparison "
                         "of people who happen to use each would not."],
                ["Assignment", "The study is within-subjects, so every participant uses both "
                               "interfaces rather than being assigned to one of them."],
                ["Why within", "Taste varies enormously between people, and that variance is far "
                               "larger than the expected effect of the interface. Between-subjects "
                               "would need a great many more participants to see through it."],
                ["Order control", "Counterbalanced, alternating across participants. Within-subjects "
                                  "makes order a confound, since whichever interface comes second "
                                  "benefits from practice and suffers from fatigue; alternating "
                                  "spreads that evenly rather than letting it load onto one design."],
                ["Other controls", "There are practice trials before each block and a break "
                                   "between the two blocks. The films themselves are drawn "
                                   "uniformly at random from the dataset."],
                ["Trials", "Each participant completes eight trials with each interface, and "
                           "the ranking sets contain ten films."],
                ["Validation", "Six further pairwise choices at the end, never used to fit w. "
                               "They are what makes the primary measure computable: a preference "
                               "vector fitted from one interface is scored on choices it has not "
                               "seen. Without them the design would name a measure the interface "
                               "could not produce."],
            ]),

            Paragraph("3.3 &nbsp; Measures", s["h2"]),
            bullets([
                "<b>Primary.</b> The share of the held-out choices that a preference vector "
                "fitted from one interface alone predicts correctly, divided by the minutes that "
                "interface consumed. Both interfaces are scored against the same held-out block, "
                "so they are compared on equal terms.",
                "<b>Secondary.</b> Time per trial; the remaining uncertainty about w; agreement "
                "of the resulting top-k recommendations between the two designs.",
                "<b>Subjective.</b> Perceived effort per interface, and which the participant "
                "felt described their taste better, both from the closing questionnaire.",
            ], s["body"]),
            Paragraph(
                "Response time is recorded per trial in the browser rather than on the server, "
                "so that it measures the participant's thinking time and not network latency.",
                s["small"]),

            Paragraph("3.4 &nbsp; Participants and recruitment", s["h2"]),
            bullets([
                "We will recruit any adult who watches films. No particular expertise is needed, "
                "because the task asks about personal taste rather than knowledge.",
                "Participants will be recruited through university mailing lists and noticeboards. "
                "The study runs online, so people can take part at a time that suits them.",
                "We will offer compensation for their time. Asking people to take part "
                "unpaid is both an ethical problem and a source of self-selection bias.",
                "The sample size will be determined from a pilot before recruitment begins "
                "rather than simply asserted, and that pilot also gives us the variance "
                "estimate needed to calculate it.",
            ], s["body"]),

            Paragraph("3.5 &nbsp; Procedure", s["h2"]),
            Paragraph(
                "Informed consent, before anything is recorded. Then instructions for the first "
                "interface, practice trials, the first block of eight, a break, instructions for "
                "the second interface, practice, the second block, the questionnaire, and a "
                "debrief explaining what was being compared and how to request erasure. Piloting "
                "with five to ten participants precedes the study proper, in person where "
                "possible so that confusion can be observed directly; pilot data is excluded from "
                "the final analysis.", s["body"]),

            Paragraph("3.6 &nbsp; Ethics and data protection", s["h2"]),
            bullets([
                "Informed consent is required before any data is recorded, and the interface "
                "cannot be started without it.",
                "Only a pseudonymous random identifier is stored. No name, no email address, no "
                "IP address.",
                "Participation is voluntary and can be ended at any point, without giving a "
                "reason and without penalty.",
                "Data minimisation: only the responses and their timings, because both are "
                "required by the stated measures and nothing else is.",
                "Right to erasure. The identifier is shown at the end so a participant can "
                "request deletion; it is the only handle linking their responses together.",
                "Review by the university ethics board before recruitment begins.",
            ], s["body"]),

            Paragraph("3.7 &nbsp; Analysis plan", s["h2"]),
            Paragraph(
                "Exclusion criteria are fixed before collection, not after: trials whose response "
                "time lies more than three standard deviations from the mean, participants whose "
                "answers show no variation, and anyone failing the attention check. Deciding these "
                "afterwards would allow the data to be filtered towards the hypothesis.", s["body"]),
            Paragraph(
                "Analysis then proceeds from descriptive statistics per condition, to a paired "
                "test on the primary measure &mdash; paired because the design is within-subjects "
                "and each participant provides both conditions &mdash; to the secondary measures "
                "with a correction for multiple comparisons, since testing several outcomes at "
                "&alpha; = 0.05 without one would make a spurious result likely.", s["body"]),

        ]

        cached = pilot.cached()
        if cached:
            story += [
                Paragraph("4 &nbsp; Pilot on simulated participants", s["h1"]),
                Paragraph(
                    "Lecture 7 frames evaluation in this field as two steps: simulated users "
                    "first, because they give full control over behaviour and real users are "
                    "costly, then human users. The study above is step two. This is step one, "
                    "and it settles two things argument cannot: whether the estimator recovers a "
                    "preference vector at all from the interactions the study can afford, and how "
                    "many participants the real study needs.", s["body"]),
                Paragraph(
                    f"{cached['n_participants']} simulated participants per time budget, each "
                    f"with a known w, answering by the Plackett-Luce model itself with beta = "
                    f"{cached['beta']} -- someone who mostly follows their own taste without being "
                    f"mechanical. Both designs receive the same time budget and spend it on as "
                    f"many trials as they can afford, at an assumed "
                    f"{cached['seconds_pairwise']:.0f}s per pairwise choice and "
                    f"{cached['seconds_ranking']:.0f}s per ranking of ten.", s["body"]),
                Table(
                    [["Budget", "Trials", "Pairwise", "Ranking", "Ceiling", "Diff", "dz", "n"]]
                    + [[f"{b['minutes']} min",
                        f"{b['pairwise_trials']}/{b['ranking_trials']}",
                        f"{b['pairwise_mean']:.1%}", f"{b['ranking_mean']:.1%}",
                        f"{b['ceiling']:.1%}", f"{b['difference']:+.1%}",
                        f"{b['dz']:+.2f}", str(b["n_required"] or "-")]
                       for b in cached["budgets"]],
                    hAlign="LEFT",
                    style=TableStyle([
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
                        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
                        ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LEFTPADDING", (0, 0), (0, -1), 0),
                    ])),
                Spacer(1, 10),
                Paragraph("4.1 &nbsp; Reading the result", s["h2"]),
                Paragraph(
                    "Agreement has to be read against the ceiling rather than against 100%. A "
                    "participant who is not perfectly consistent with themselves caps how well any "
                    "preference vector can predict their held-out choices, including their own "
                    "true one; that ceiling sits near 80%, so a design reaching 77% has recovered "
                    "most of what was recoverable.", s["body"]),
                Paragraph(
                    "Ranking leads at every budget, but only barely at the shortest. At two "
                    "minutes a ranking block affords two trials, too few to constrain "
                    f"{cached['n_features']} weights, and the advantage nearly vanishes. This is a "
                    "design finding in its own right: too short a session does not merely weaken "
                    "the study, it erases the effect it is trying to measure.", s["body"]),
                Paragraph(
                    "The effect is small. Where ranking leads it leads by one to three percentage "
                    "points, and the sample sizes that follow range from about 45 to over a "
                    "thousand. That spread is itself informative: with an effect this small, even "
                    f"{cached['n_participants']} simulated participants cannot pin the required n "
                    "tightly. The defensible reading is that the study needs on the order of one "
                    "to two hundred participants rather than the twenty a coursework study would "
                    "otherwise assume, and that quoting the most flattering row would be a "
                    "mistake.", s["body"]),
                Paragraph("4.2 &nbsp; What the timing assumption is carrying", s["h2"]),
                Paragraph(
                    "The comparison is normalised by time, and a simulation cannot know how long "
                    "a trial takes, so the durations are assumptions. Varying the assumed time "
                    "for a ranking shows the direction holds while a ranking takes up to about a "
                    "minute and disappears at ninety seconds, where only four rankings fit the "
                    "budget. The conclusion therefore depends on the assumption, but not "
                    "delicately: it fails only in a regime where the ranking block is too short "
                    "to be informative anyway. The interface records real per-trial times, so a "
                    "human pilot replaces this assumption with measurement.", s["body"]),
                Paragraph(
                    "None of this is evidence about people. Every participant here follows the "
                    "model exactly, which is the one thing a real participant certainly does not "
                    "do. The pilot establishes that the instrument works and what sample the "
                    "study needs; it cannot establish that the hypothesis is true.", s["small"]),
            ]

        story += [
            Spacer(1, 14),
            Paragraph(
                "The interface described in section 3 is implemented and reachable from the "
                "project landing page. This report is generated from the implementation by "
                "<b>manage.py build_project4_report</b>, so the feature list, dataset size and "
                "prior strength quoted above cannot drift from the code.", s["small"]),
        ]

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        SimpleDocTemplate(
            str(OUTPUT), pagesize=A4,
            leftMargin=2.4 * cm, rightMargin=2.4 * cm,
            topMargin=2.2 * cm, bottomMargin=2.2 * cm,
            title="Project 4 - Preference Elicitation",
            author="HCAI Group 29",
        ).build(story)

        self.stdout.write(self.style.SUCCESS(
            f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)"
        ))

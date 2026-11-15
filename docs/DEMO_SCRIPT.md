# 5-minute demo script (≈ 10 questions)

> Written against the **synthetic test data**. Once the real Kaggle dataset is ingested, re-run each question,
> check the plan numbers exist, and update the spoken numbers. Say clearly on camera which dataset is loaded.

**0:00 – 0:30 Intro.** "This assistant answers questions about floor plans by querying a SQLite database built from
segmentation masks. It never answers from memory: each answer shows its SQL and result rows." Show the Overview tab and
the *synthetic test data* banner (or the real-dataset banner-free view).

**0:30 – 1:15 Data.** Dataset tab: counts of plans/rooms/doors/windows. Floor plan tab: pick Plan 3, show the original
next to the overlay (rooms boxed, doors orange, windows blue). "Areas are in pixels – there is no scale in the data."

**1:15 – 4:15 Questions (Ask tab, expand "SQL query and evidence" each time):**
1. "How many rooms does plan 3 have?" → 6
2. "What is the largest room in plan 7?" → living room #3, 17,064 px
3. "Which plans have more than 3 doors?" → plans 2, 3, 5, 6, 7, 8
4. "Which room in plan 4 has no windows?" → hallway #2
5. "What is the average number of rooms across all plans?" → 4.75
6. "How many bedrooms are in plan 6?" → 3
7. "Which plan has the most rooms?" → plan 6
8. "Which plans have more than 20 doors?" → none (empty result stated plainly)
9. "How many rooms does plan 12 have?" → declines: plan 12 does not exist (1–8)
10. **"What is the area of the living room in plan 2 in square metres?" → DECLINES**: no scale in the data, only pixels.

**4:15 – 4:45 Evaluation tab.** Run the 28-question evaluation: total, correct, accuracy, failure cases. State that the
synthetic score is a pipeline check and is not a real-dataset result.

**4:45 – 5:00 Wrap-up.** Safety (read-only SQL, validator + authorizer + timeout), LLM optional, limitations, next steps.

# Healthcare business glossary

Internal definitions used in hospital reporting for the admissions table (`healthcare_dataset.csv`).
Each `##` section is one retrievable chunk; the text before the dash is the chunk ID.
All rules are stated in terms of the table's real columns.

## hc_billable_days — Billable days
The number of days an admission is billed for. Both the admission day and the discharge day count, so
billable days = (`Discharge Date` − `Date of Admission`) in days + 1. A patient admitted and discharged
one day later has 2 billable days.

## hc_daily_rate — Daily rate
`Billing Amount` divided by the admission's billable days (discharge minus admission, plus one).
"Average daily rate" means the mean of the per-admission daily rates, not total billing divided by total days.

## hc_long_stay — Long-stay admission
An admission whose `Discharge Date` is at least 21 days after its `Date of Admission`.
Long-stay admissions are reviewed by the utilization committee.

## hc_short_stay — Short stay
An admission discharged no more than 3 days after its `Date of Admission` (discharge minus admission ≤ 3).

## hc_high_cost — High-cost admission
A single admission whose `Billing Amount` is greater than 42,500.

## hc_cost_overrun — Cost overrun
An admission billed at more than 1.5 times the average `Billing Amount` of all admissions with the same
`Medical Condition`.

## hc_premium_insurer — Premium insurer
`Insurance Provider` is Aetna or Cigna. All other providers are standard insurers.

## hc_government_payer — Government payer
`Insurance Provider` is Medicare or UnitedHealthcare. UnitedHealthcare counts because the hospital's
managed-Medicaid contract is administered through it.

## hc_senior_patient — Senior patient
A patient whose `Age` is 67 or older (the hospital aligns with the full retirement age, not 65).

## hc_young_adult — Young adult
A patient whose `Age` is between 18 and 34 inclusive.

## hc_icu_room — ICU room
`Room Number` from 400 to 449 inclusive. These rooms are the intensive care unit.

## hc_step_down_unit — Step-down unit
`Room Number` from 450 to 500 inclusive: the step-down (intermediate care) unit next to the ICU.

## hc_chronic_condition — Chronic condition
`Medical Condition` is Diabetes, Hypertension, Obesity, or Arthritis. Asthma and Cancer are tracked in
separate programs and are NOT counted as chronic conditions in reporting.

## hc_unplanned_admission — Unplanned admission
`Admission Type` is Emergency or Urgent. Elective admissions are planned.

## hc_flagged_result — Flagged result
`Test Results` is Abnormal or Inconclusive. Both require a follow-up appointment.

## hc_fiscal_year — Fiscal year
The hospital fiscal year runs from July 1 to June 30 and is named after the calendar year in which it ends,
based on `Date of Admission`. FY2022 = admissions from 2021-07-01 through 2022-06-30.

## hc_fiscal_quarter — Fiscal quarter
Quarters of the July–June fiscal year, by `Date of Admission` month: FQ1 = July–September,
FQ2 = October–December, FQ3 = January–March, FQ4 = April–June.

## hc_flu_season — Flu season
Admissions whose `Date of Admission` falls in November, December, January, or February (any year).

## hc_formulary_drug — Formulary drug
`Medication` is Lipitor, Penicillin, or Ibuprofen: the drugs on the hospital's preferred formulary.
Aspirin and Paracetamol are non-formulary.

## hc_rare_blood_type — Rare blood type
`Blood Type` is AB-, B-, or A-. The blood bank keeps a separate reserve for these three types.

## hc_high_risk_patient — High-risk patient
A patient aged 60 or older (`Age` ≥ 60) whose `Medical Condition` is Cancer or Diabetes.

## hc_high_cost_condition — High-cost condition
A `Medical Condition` whose average `Billing Amount` across all admissions is above 25,500.
This describes a condition, not an individual admission.

## hc_senior_care_tier — Senior care tier
Patients with `Age` 80 or older, who are routed to the geriatric ward. Narrower than "senior patient".

## hc_billing_quarter — Billing quarter
The calendar quarter (Jan–Mar = Q1 ... Oct–Dec = Q4) of the `Discharge Date`, when the claim is filed.
Not the same as the fiscal quarter.

## hc_flagged_claim — Flagged claim
An admission with `Billing Amount` below 2,000, which is sent for a manual billing audit.
Unrelated to flagged test results.

## hc_premium_room — Premium room
`Room Number` 480 or higher: the private suites. Unrelated to premium insurers.

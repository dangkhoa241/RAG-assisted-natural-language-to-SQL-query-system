"""Build eval/sql_benchmark/example_bank.jsonl: question -> SQL pairs used for retrieval.

The bank deliberately uses schemas that do NOT appear in the benchmark (HR, school, logistics,
hotel, energy), so retrieval can only teach SQL patterns, never leak benchmark answers.
Every SQL statement is executed against a small synthetic table with the same columns,
so a typo in the bank fails here instead of silently teaching the LLM a broken pattern.

Usage (from the repo root):  python eval/sql_benchmark/build_example_bank.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent / "example_bank.jsonl"

# column -> ("cat", values) | ("int", lo, hi) | ("real", lo, hi) | ("date", start, end) | ("text", prefix)
SCHEMAS = {
    "hr_employees": {
        "Employee ID": ("int", 1, 5000),
        "Full Name": ("text", "Employee"),
        "Department": ("cat", ["Engineering", "Sales", "Marketing", "Finance", "HR", "Support"]),
        "Job Level": ("cat", ["Junior", "Mid", "Senior", "Lead"]),
        "Salary": ("real", 40000, 180000),
        "Bonus": ("real", 0, 30000),
        "Hire Date": ("date", "2015-01-01", "2024-12-31"),
        "Performance Score": ("int", 1, 5),
        "Remote": ("cat", ["Yes", "No"]),
        "Office": ("cat", ["Austin", "Denver", "Boston", "Seattle"]),
        "Years Experience": ("int", 0, 30),
    },
    "school_students": {
        "Student ID": ("int", 1, 3000),
        "Student Name": ("text", "Student"),
        "Grade Level": ("int", 9, 12),
        "Track": ("cat", ["STEM", "Arts", "Humanities", "Business"]),
        "School": ("cat", ["Lincoln High", "Roosevelt High", "Kennedy High"]),
        "GPA": ("real", 1.5, 4.0),
        "Attendance Rate": ("real", 0.6, 1.0),
        "Math Score": ("int", 40, 100),
        "Reading Score": ("int", 40, 100),
        "Scholarship": ("cat", ["Yes", "No"]),
        "Enrollment Date": ("date", "2019-08-15", "2024-09-01"),
    },
    "logistics_shipments": {
        "Shipment ID": ("int", 1, 9000),
        "Ship Date": ("date", "2022-01-01", "2024-12-31"),
        "Delivery Date": ("date", "2022-01-03", "2025-01-10"),
        "Carrier": ("cat", ["DHL", "FedEx", "UPS", "USPS"]),
        "Origin City": ("cat", ["Chicago", "Dallas", "Atlanta", "Phoenix"]),
        "Destination City": ("cat", ["Miami", "Seattle", "Denver", "New York"]),
        "Priority": ("cat", ["Standard", "Express", "Overnight"]),
        "Status": ("cat", ["Delivered", "In Transit", "Delayed", "Lost"]),
        "Weight Kg": ("real", 0.5, 80),
        "Shipping Cost": ("real", 5, 400),
    },
    "hotel_bookings": {
        "Booking ID": ("int", 1, 8000),
        "Guest Name": ("text", "Guest"),
        "Check In Date": ("date", "2022-01-01", "2024-12-20"),
        "Check Out Date": ("date", "2022-01-02", "2024-12-31"),
        "Room Type": ("cat", ["Standard", "Deluxe", "Suite"]),
        "Booking Source": ("cat", ["Website", "Travel Agent", "Walk-in"]),
        "Country": ("cat", ["USA", "Canada", "Mexico", "France", "Japan"]),
        "Guests": ("int", 1, 5),
        "Nightly Rate": ("real", 80, 600),
        "Total Price": ("real", 80, 5000),
        "Cancelled": ("cat", ["Yes", "No"]),
        "Review Score": ("int", 1, 10),
    },
    "energy_meters": {
        "Reading ID": ("int", 1, 20000),
        "Reading Date": ("date", "2021-01-01", "2024-12-31"),
        "Meter Type": ("cat", ["Residential", "Commercial", "Industrial"]),
        "State": ("cat", ["CA", "TX", "NY", "FL"]),
        "Tariff Plan": ("cat", ["Flat", "Time-of-Use", "Tiered"]),
        "Solar Panels": ("cat", ["Yes", "No"]),
        "Usage kWh": ("real", 50, 5000),
        "Peak Demand kW": ("real", 1, 400),
        "Bill Amount": ("real", 20, 2500),
        "Outage Minutes": ("int", 0, 600),
    },
}

# (intent, question, sql) per schema: 6 per intent, 30 per schema.
PAIRS = {
    "hr_employees": [
        ("filter", "show engineers hired after 2022", "SELECT * FROM data WHERE `Department` = 'Engineering' AND `Hire Date` > '2022-12-31'"),
        ("filter", "list remote senior employees in Denver", "SELECT * FROM data WHERE `Remote` = 'Yes' AND `Job Level` = 'Senior' AND `Office` = 'Denver'"),
        ("filter", "the 10 highest paid employees", "SELECT * FROM data ORDER BY `Salary` DESC LIMIT 10"),
        ("filter", "employees with a performance score of 5 and a bonus under 1000", "SELECT * FROM data WHERE `Performance Score` = 5 AND `Bonus` < 1000"),
        ("filter", "who joined in March 2021", "SELECT * FROM data WHERE strftime('%Y-%m', `Hire Date`) = '2021-03'"),
        ("filter", "staff with between 10 and 15 years of experience in finance", "SELECT * FROM data WHERE `Years Experience` BETWEEN 10 AND 15 AND `Department` = 'Finance'"),
        ("count", "how many employees work remotely", "SELECT COUNT(*) FROM data WHERE `Remote` = 'Yes'"),
        ("count", "number of leads in the sales department", "SELECT COUNT(*) FROM data WHERE `Job Level` = 'Lead' AND `Department` = 'Sales'"),
        ("count", "how many people were hired in 2023", "SELECT COUNT(*) FROM data WHERE strftime('%Y', `Hire Date`) = '2023'"),
        ("count", "how many offices do we have", "SELECT COUNT(DISTINCT `Office`) FROM data"),
        ("count", "count employees earning more than 150000", "SELECT COUNT(*) FROM data WHERE `Salary` > 150000"),
        ("count", "how many employees have worked here more than 5 years", "SELECT COUNT(*) FROM data WHERE julianday('now') - julianday(`Hire Date`) > 5 * 365"),
        ("aggregate", "average salary by department", "SELECT `Department`, AVG(`Salary`) FROM data GROUP BY `Department`"),
        ("aggregate", "total bonus paid per office", "SELECT `Office`, SUM(`Bonus`) FROM data GROUP BY `Office`"),
        ("aggregate", "highest salary at each job level", "SELECT `Job Level`, MAX(`Salary`) FROM data GROUP BY `Job Level`"),
        ("aggregate", "average performance score of marketing staff", "SELECT AVG(`Performance Score`) FROM data WHERE `Department` = 'Marketing'"),
        ("aggregate", "minimum years of experience for each department among seniors", "SELECT `Department`, MIN(`Years Experience`) FROM data WHERE `Job Level` = 'Senior' GROUP BY `Department`"),
        ("aggregate", "what is the total payroll", "SELECT SUM(`Salary`) FROM data"),
        ("compare", "do remote employees earn more than office employees on average", "SELECT `Remote`, AVG(`Salary`) FROM data GROUP BY `Remote`"),
        ("compare", "compare average bonus between sales and marketing", "SELECT `Department`, AVG(`Bonus`) FROM data WHERE `Department` IN ('Sales', 'Marketing') GROUP BY `Department`"),
        ("compare", "Austin vs Seattle headcount", "SELECT `Office`, COUNT(*) FROM data WHERE `Office` IN ('Austin', 'Seattle') GROUP BY `Office`"),
        ("compare", "are juniors or mids rated higher on average", "SELECT `Job Level`, AVG(`Performance Score`) FROM data WHERE `Job Level` IN ('Junior', 'Mid') GROUP BY `Job Level`"),
        ("compare", "which department has the bigger total payroll, engineering or finance", "SELECT `Department`, SUM(`Salary`) FROM data WHERE `Department` IN ('Engineering', 'Finance') GROUP BY `Department`"),
        ("compare", "compare hires in 2022 and 2023", "SELECT strftime('%Y', `Hire Date`), COUNT(*) FROM data WHERE strftime('%Y', `Hire Date`) IN ('2022', '2023') GROUP BY 1"),
        ("trend", "hires per year", "SELECT strftime('%Y', `Hire Date`), COUNT(*) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "monthly hiring in 2024", "SELECT strftime('%Y-%m', `Hire Date`), COUNT(*) FROM data WHERE strftime('%Y', `Hire Date`) = '2024' GROUP BY 1 ORDER BY 1"),
        ("trend", "average starting salary by hire year", "SELECT strftime('%Y', `Hire Date`), AVG(`Salary`) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "how has engineering hiring changed year over year", "SELECT strftime('%Y', `Hire Date`), COUNT(*) FROM data WHERE `Department` = 'Engineering' GROUP BY 1 ORDER BY 1"),
        ("trend", "total bonus by hire month for 2023 hires", "SELECT strftime('%Y-%m', `Hire Date`), SUM(`Bonus`) FROM data WHERE strftime('%Y', `Hire Date`) = '2023' GROUP BY 1 ORDER BY 1"),
        ("trend", "trend of remote hires by year", "SELECT strftime('%Y', `Hire Date`), COUNT(*) FROM data WHERE `Remote` = 'Yes' GROUP BY 1 ORDER BY 1"),
    ],
    "school_students": [
        ("filter", "show STEM students with a GPA above 3.8", "SELECT * FROM data WHERE `Track` = 'STEM' AND `GPA` > 3.8"),
        ("filter", "list seniors at Lincoln High on scholarship", "SELECT * FROM data WHERE `Grade Level` = 12 AND `School` = 'Lincoln High' AND `Scholarship` = 'Yes'"),
        ("filter", "students with attendance under 70 percent", "SELECT * FROM data WHERE `Attendance Rate` < 0.7"),
        ("filter", "the 5 best math scores", "SELECT * FROM data ORDER BY `Math Score` DESC LIMIT 5"),
        ("filter", "students who enrolled in fall 2023", "SELECT * FROM data WHERE `Enrollment Date` BETWEEN '2023-08-01' AND '2023-12-31'"),
        ("filter", "arts students who scored below 50 in reading", "SELECT * FROM data WHERE `Track` = 'Arts' AND `Reading Score` < 50"),
        ("count", "how many students are on scholarship", "SELECT COUNT(*) FROM data WHERE `Scholarship` = 'Yes'"),
        ("count", "number of freshmen at Kennedy High", "SELECT COUNT(*) FROM data WHERE `Grade Level` = 9 AND `School` = 'Kennedy High'"),
        ("count", "how many students have a GPA of at least 3.5", "SELECT COUNT(*) FROM data WHERE `GPA` >= 3.5"),
        ("count", "how many tracks are offered", "SELECT COUNT(DISTINCT `Track`) FROM data"),
        ("count", "how many students enrolled in 2022", "SELECT COUNT(*) FROM data WHERE strftime('%Y', `Enrollment Date`) = '2022'"),
        ("count", "count business track students with perfect attendance", "SELECT COUNT(*) FROM data WHERE `Track` = 'Business' AND `Attendance Rate` = 1.0"),
        ("aggregate", "average GPA by school", "SELECT `School`, AVG(`GPA`) FROM data GROUP BY `School`"),
        ("aggregate", "average math score per track", "SELECT `Track`, AVG(`Math Score`) FROM data GROUP BY `Track`"),
        ("aggregate", "lowest attendance rate in each grade", "SELECT `Grade Level`, MIN(`Attendance Rate`) FROM data GROUP BY `Grade Level`"),
        ("aggregate", "top reading score for each school", "SELECT `School`, MAX(`Reading Score`) FROM data GROUP BY `School`"),
        ("aggregate", "mean GPA of scholarship students", "SELECT AVG(`GPA`) FROM data WHERE `Scholarship` = 'Yes'"),
        ("aggregate", "average reading score by track for juniors", "SELECT `Track`, AVG(`Reading Score`) FROM data WHERE `Grade Level` = 11 GROUP BY `Track`"),
        ("compare", "do scholarship students have higher GPAs than others", "SELECT `Scholarship`, AVG(`GPA`) FROM data GROUP BY `Scholarship`"),
        ("compare", "Lincoln vs Roosevelt average math score", "SELECT `School`, AVG(`Math Score`) FROM data WHERE `School` IN ('Lincoln High', 'Roosevelt High') GROUP BY `School`"),
        ("compare", "which track has better attendance, arts or humanities", "SELECT `Track`, AVG(`Attendance Rate`) FROM data WHERE `Track` IN ('Arts', 'Humanities') GROUP BY `Track`"),
        ("compare", "compare the number of STEM and business students", "SELECT `Track`, COUNT(*) FROM data WHERE `Track` IN ('STEM', 'Business') GROUP BY `Track`"),
        ("compare", "are 9th graders or 12th graders better at reading on average", "SELECT `Grade Level`, AVG(`Reading Score`) FROM data WHERE `Grade Level` IN (9, 12) GROUP BY `Grade Level`"),
        ("compare", "highest GPA at Kennedy versus Lincoln", "SELECT `School`, MAX(`GPA`) FROM data WHERE `School` IN ('Kennedy High', 'Lincoln High') GROUP BY `School`"),
        ("trend", "enrollments per year", "SELECT strftime('%Y', `Enrollment Date`), COUNT(*) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "monthly enrollments in 2023", "SELECT strftime('%Y-%m', `Enrollment Date`), COUNT(*) FROM data WHERE strftime('%Y', `Enrollment Date`) = '2023' GROUP BY 1 ORDER BY 1"),
        ("trend", "average GPA by enrollment year", "SELECT strftime('%Y', `Enrollment Date`), AVG(`GPA`) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "how has STEM enrollment changed each year", "SELECT strftime('%Y', `Enrollment Date`), COUNT(*) FROM data WHERE `Track` = 'STEM' GROUP BY 1 ORDER BY 1"),
        ("trend", "scholarship awards by enrollment year", "SELECT strftime('%Y', `Enrollment Date`), COUNT(*) FROM data WHERE `Scholarship` = 'Yes' GROUP BY 1 ORDER BY 1"),
        ("trend", "average attendance by enrollment month in 2022", "SELECT strftime('%Y-%m', `Enrollment Date`), AVG(`Attendance Rate`) FROM data WHERE strftime('%Y', `Enrollment Date`) = '2022' GROUP BY 1 ORDER BY 1"),
    ],
    "logistics_shipments": [
        ("filter", "show delayed FedEx shipments", "SELECT * FROM data WHERE `Carrier` = 'FedEx' AND `Status` = 'Delayed'"),
        ("filter", "overnight shipments heavier than 50 kg", "SELECT * FROM data WHERE `Priority` = 'Overnight' AND `Weight Kg` > 50"),
        ("filter", "shipments that took more than 10 days to deliver", "SELECT * FROM data WHERE julianday(`Delivery Date`) - julianday(`Ship Date`) > 10"),
        ("filter", "the 3 most expensive shipments to Miami", "SELECT * FROM data WHERE `Destination City` = 'Miami' ORDER BY `Shipping Cost` DESC LIMIT 3"),
        ("filter", "list lost packages from Chicago", "SELECT * FROM data WHERE `Status` = 'Lost' AND `Origin City` = 'Chicago'"),
        ("filter", "shipments sent on 2024-07-04", "SELECT * FROM data WHERE `Ship Date` = '2024-07-04'"),
        ("count", "how many shipments were lost", "SELECT COUNT(*) FROM data WHERE `Status` = 'Lost'"),
        ("count", "number of express shipments from Dallas", "SELECT COUNT(*) FROM data WHERE `Priority` = 'Express' AND `Origin City` = 'Dallas'"),
        ("count", "how many shipments cost over 300", "SELECT COUNT(*) FROM data WHERE `Shipping Cost` > 300"),
        ("count", "how many carriers do we use", "SELECT COUNT(DISTINCT `Carrier`) FROM data"),
        ("count", "how many shipments went out in June 2023", "SELECT COUNT(*) FROM data WHERE strftime('%Y-%m', `Ship Date`) = '2023-06'"),
        ("count", "count UPS deliveries to Seattle", "SELECT COUNT(*) FROM data WHERE `Carrier` = 'UPS' AND `Destination City` = 'Seattle'"),
        ("aggregate", "average shipping cost by carrier", "SELECT `Carrier`, AVG(`Shipping Cost`) FROM data GROUP BY `Carrier`"),
        ("aggregate", "total weight shipped per origin city", "SELECT `Origin City`, SUM(`Weight Kg`) FROM data GROUP BY `Origin City`"),
        ("aggregate", "average delivery time in days by priority", "SELECT `Priority`, AVG(julianday(`Delivery Date`) - julianday(`Ship Date`)) FROM data GROUP BY `Priority`"),
        ("aggregate", "heaviest shipment for each destination", "SELECT `Destination City`, MAX(`Weight Kg`) FROM data GROUP BY `Destination City`"),
        ("aggregate", "total shipping spend with DHL", "SELECT SUM(`Shipping Cost`) FROM data WHERE `Carrier` = 'DHL'"),
        ("aggregate", "cheapest shipping cost per carrier for express shipments", "SELECT `Carrier`, MIN(`Shipping Cost`) FROM data WHERE `Priority` = 'Express' GROUP BY `Carrier`"),
        ("compare", "is UPS or USPS cheaper on average", "SELECT `Carrier`, AVG(`Shipping Cost`) FROM data WHERE `Carrier` IN ('UPS', 'USPS') GROUP BY `Carrier`"),
        ("compare", "compare delays between DHL and FedEx", "SELECT `Carrier`, COUNT(*) FROM data WHERE `Status` = 'Delayed' AND `Carrier` IN ('DHL', 'FedEx') GROUP BY `Carrier`"),
        ("compare", "do express shipments arrive faster than standard ones", "SELECT `Priority`, AVG(julianday(`Delivery Date`) - julianday(`Ship Date`)) FROM data WHERE `Priority` IN ('Express', 'Standard') GROUP BY `Priority`"),
        ("compare", "Atlanta vs Phoenix total shipping cost", "SELECT `Origin City`, SUM(`Shipping Cost`) FROM data WHERE `Origin City` IN ('Atlanta', 'Phoenix') GROUP BY `Origin City`"),
        ("compare", "which gets heavier packages on average, Denver or New York", "SELECT `Destination City`, AVG(`Weight Kg`) FROM data WHERE `Destination City` IN ('Denver', 'New York') GROUP BY `Destination City`"),
        ("compare", "shipment volume in 2023 compared with 2024", "SELECT strftime('%Y', `Ship Date`), COUNT(*) FROM data WHERE strftime('%Y', `Ship Date`) IN ('2023', '2024') GROUP BY 1"),
        ("trend", "shipments per month in 2024", "SELECT strftime('%Y-%m', `Ship Date`), COUNT(*) FROM data WHERE strftime('%Y', `Ship Date`) = '2024' GROUP BY 1 ORDER BY 1"),
        ("trend", "total shipping cost by year", "SELECT strftime('%Y', `Ship Date`), SUM(`Shipping Cost`) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "average delivery time by year", "SELECT strftime('%Y', `Ship Date`), AVG(julianday(`Delivery Date`) - julianday(`Ship Date`)) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "how did lost shipments change year over year", "SELECT strftime('%Y', `Ship Date`), COUNT(*) FROM data WHERE `Status` = 'Lost' GROUP BY 1 ORDER BY 1"),
        ("trend", "monthly FedEx spend in 2023", "SELECT strftime('%Y-%m', `Ship Date`), SUM(`Shipping Cost`) FROM data WHERE `Carrier` = 'FedEx' AND strftime('%Y', `Ship Date`) = '2023' GROUP BY 1 ORDER BY 1"),
        ("trend", "average package weight per month in 2022", "SELECT strftime('%Y-%m', `Ship Date`), AVG(`Weight Kg`) FROM data WHERE strftime('%Y', `Ship Date`) = '2022' GROUP BY 1 ORDER BY 1"),
    ],
    "hotel_bookings": [
        ("filter", "show cancelled suite bookings", "SELECT * FROM data WHERE `Cancelled` = 'Yes' AND `Room Type` = 'Suite'"),
        ("filter", "bookings from Japan with more than 3 guests", "SELECT * FROM data WHERE `Country` = 'Japan' AND `Guests` > 3"),
        ("filter", "stays longer than 7 nights", "SELECT * FROM data WHERE julianday(`Check Out Date`) - julianday(`Check In Date`) > 7"),
        ("filter", "the 5 cheapest deluxe bookings", "SELECT * FROM data WHERE `Room Type` = 'Deluxe' ORDER BY `Total Price` ASC LIMIT 5"),
        ("filter", "walk-in guests who left a review score below 4", "SELECT * FROM data WHERE `Booking Source` = 'Walk-in' AND `Review Score` < 4"),
        ("filter", "check-ins during August 2024", "SELECT * FROM data WHERE strftime('%Y-%m', `Check In Date`) = '2024-08'"),
        ("count", "how many bookings were cancelled", "SELECT COUNT(*) FROM data WHERE `Cancelled` = 'Yes'"),
        ("count", "number of suite bookings made through travel agents", "SELECT COUNT(*) FROM data WHERE `Room Type` = 'Suite' AND `Booking Source` = 'Travel Agent'"),
        ("count", "how many guests came from Canada in 2023", "SELECT COUNT(*) FROM data WHERE `Country` = 'Canada' AND strftime('%Y', `Check In Date`) = '2023'"),
        ("count", "how many countries do our guests come from", "SELECT COUNT(DISTINCT `Country`) FROM data"),
        ("count", "how many bookings had a nightly rate of at least 400", "SELECT COUNT(*) FROM data WHERE `Nightly Rate` >= 400"),
        ("count", "count stays of exactly one night", "SELECT COUNT(*) FROM data WHERE julianday(`Check Out Date`) - julianday(`Check In Date`) = 1"),
        ("aggregate", "average nightly rate by room type", "SELECT `Room Type`, AVG(`Nightly Rate`) FROM data GROUP BY `Room Type`"),
        ("aggregate", "total revenue per booking source", "SELECT `Booking Source`, SUM(`Total Price`) FROM data GROUP BY `Booking Source`"),
        ("aggregate", "average review score by country", "SELECT `Country`, AVG(`Review Score`) FROM data GROUP BY `Country`"),
        ("aggregate", "most expensive booking for each room type", "SELECT `Room Type`, MAX(`Total Price`) FROM data GROUP BY `Room Type`"),
        ("aggregate", "average length of stay in nights", "SELECT AVG(julianday(`Check Out Date`) - julianday(`Check In Date`)) FROM data"),
        ("aggregate", "total revenue from non-cancelled bookings by country", "SELECT `Country`, SUM(`Total Price`) FROM data WHERE `Cancelled` = 'No' GROUP BY `Country`"),
        ("compare", "do suites get better reviews than standard rooms", "SELECT `Room Type`, AVG(`Review Score`) FROM data WHERE `Room Type` IN ('Suite', 'Standard') GROUP BY `Room Type`"),
        ("compare", "website vs travel agent total revenue", "SELECT `Booking Source`, SUM(`Total Price`) FROM data WHERE `Booking Source` IN ('Website', 'Travel Agent') GROUP BY `Booking Source`"),
        ("compare", "are cancelled bookings pricier than completed ones on average", "SELECT `Cancelled`, AVG(`Total Price`) FROM data GROUP BY `Cancelled`"),
        ("compare", "compare the number of bookings from France and Mexico", "SELECT `Country`, COUNT(*) FROM data WHERE `Country` IN ('France', 'Mexico') GROUP BY `Country`"),
        ("compare", "who stays longer on average, USA or Japan guests", "SELECT `Country`, AVG(julianday(`Check Out Date`) - julianday(`Check In Date`)) FROM data WHERE `Country` IN ('USA', 'Japan') GROUP BY `Country`"),
        ("compare", "deluxe or suite: which has the higher maximum nightly rate", "SELECT `Room Type`, MAX(`Nightly Rate`) FROM data WHERE `Room Type` IN ('Deluxe', 'Suite') GROUP BY `Room Type`"),
        ("trend", "bookings per month in 2023", "SELECT strftime('%Y-%m', `Check In Date`), COUNT(*) FROM data WHERE strftime('%Y', `Check In Date`) = '2023' GROUP BY 1 ORDER BY 1"),
        ("trend", "yearly room revenue from all bookings", "SELECT strftime('%Y', `Check In Date`), SUM(`Total Price`) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "average nightly rate per month in 2024", "SELECT strftime('%Y-%m', `Check In Date`), AVG(`Nightly Rate`) FROM data WHERE strftime('%Y', `Check In Date`) = '2024' GROUP BY 1 ORDER BY 1"),
        ("trend", "how have cancellations changed each year", "SELECT strftime('%Y', `Check In Date`), COUNT(*) FROM data WHERE `Cancelled` = 'Yes' GROUP BY 1 ORDER BY 1"),
        ("trend", "yearly average review score for suites", "SELECT strftime('%Y', `Check In Date`), AVG(`Review Score`) FROM data WHERE `Room Type` = 'Suite' GROUP BY 1 ORDER BY 1"),
        ("trend", "monthly walk-in bookings in 2022", "SELECT strftime('%Y-%m', `Check In Date`), COUNT(*) FROM data WHERE `Booking Source` = 'Walk-in' AND strftime('%Y', `Check In Date`) = '2022' GROUP BY 1 ORDER BY 1"),
    ],
    "energy_meters": [
        ("filter", "show commercial meters in Texas with outages over 120 minutes", "SELECT * FROM data WHERE `Meter Type` = 'Commercial' AND `State` = 'TX' AND `Outage Minutes` > 120"),
        ("filter", "readings with solar panels and a bill under 50", "SELECT * FROM data WHERE `Solar Panels` = 'Yes' AND `Bill Amount` < 50"),
        ("filter", "the 10 largest peak demand readings", "SELECT * FROM data ORDER BY `Peak Demand kW` DESC LIMIT 10"),
        ("filter", "industrial readings from New York on the tiered plan", "SELECT * FROM data WHERE `Meter Type` = 'Industrial' AND `State` = 'NY' AND `Tariff Plan` = 'Tiered'"),
        ("filter", "readings taken in the first week of January 2024", "SELECT * FROM data WHERE `Reading Date` BETWEEN '2024-01-01' AND '2024-01-07'"),
        ("filter", "residential usage between 1000 and 1500 kWh", "SELECT * FROM data WHERE `Meter Type` = 'Residential' AND `Usage kWh` BETWEEN 1000 AND 1500"),
        ("count", "how many readings had no outage", "SELECT COUNT(*) FROM data WHERE `Outage Minutes` = 0"),
        ("count", "number of residential meters with solar panels in California", "SELECT COUNT(*) FROM data WHERE `Meter Type` = 'Residential' AND `Solar Panels` = 'Yes' AND `State` = 'CA'"),
        ("count", "how many readings were taken in 2022", "SELECT COUNT(*) FROM data WHERE strftime('%Y', `Reading Date`) = '2022'"),
        ("count", "how many tariff plans exist", "SELECT COUNT(DISTINCT `Tariff Plan`) FROM data"),
        ("count", "how many bills were above 2000", "SELECT COUNT(*) FROM data WHERE `Bill Amount` > 2000"),
        ("count", "count Florida readings on time-of-use pricing", "SELECT COUNT(*) FROM data WHERE `State` = 'FL' AND `Tariff Plan` = 'Time-of-Use'"),
        ("aggregate", "average usage by meter type", "SELECT `Meter Type`, AVG(`Usage kWh`) FROM data GROUP BY `Meter Type`"),
        ("aggregate", "total billed per state", "SELECT `State`, SUM(`Bill Amount`) FROM data GROUP BY `State`"),
        ("aggregate", "maximum outage minutes per tariff plan", "SELECT `Tariff Plan`, MAX(`Outage Minutes`) FROM data GROUP BY `Tariff Plan`"),
        ("aggregate", "average bill for homes with solar panels", "SELECT AVG(`Bill Amount`) FROM data WHERE `Solar Panels` = 'Yes' AND `Meter Type` = 'Residential'"),
        ("aggregate", "lowest peak demand for each state", "SELECT `State`, MIN(`Peak Demand kW`) FROM data GROUP BY `State`"),
        ("aggregate", "total energy used in Texas", "SELECT SUM(`Usage kWh`) FROM data WHERE `State` = 'TX'"),
        ("compare", "do solar homes pay less than non-solar homes on average", "SELECT `Solar Panels`, AVG(`Bill Amount`) FROM data WHERE `Meter Type` = 'Residential' GROUP BY `Solar Panels`"),
        ("compare", "California vs Texas average usage", "SELECT `State`, AVG(`Usage kWh`) FROM data WHERE `State` IN ('CA', 'TX') GROUP BY `State`"),
        ("compare", "which plan has more outage minutes in total, flat or tiered", "SELECT `Tariff Plan`, SUM(`Outage Minutes`) FROM data WHERE `Tariff Plan` IN ('Flat', 'Tiered') GROUP BY `Tariff Plan`"),
        ("compare", "compare the number of commercial and industrial readings", "SELECT `Meter Type`, COUNT(*) FROM data WHERE `Meter Type` IN ('Commercial', 'Industrial') GROUP BY `Meter Type`"),
        ("compare", "is peak demand higher in New York or Florida on average", "SELECT `State`, AVG(`Peak Demand kW`) FROM data WHERE `State` IN ('NY', 'FL') GROUP BY `State`"),
        ("compare", "total usage in 2023 versus 2024", "SELECT strftime('%Y', `Reading Date`), SUM(`Usage kWh`) FROM data WHERE strftime('%Y', `Reading Date`) IN ('2023', '2024') GROUP BY 1"),
        ("trend", "monthly usage in 2024", "SELECT strftime('%Y-%m', `Reading Date`), SUM(`Usage kWh`) FROM data WHERE strftime('%Y', `Reading Date`) = '2024' GROUP BY 1 ORDER BY 1"),
        ("trend", "average bill by year", "SELECT strftime('%Y', `Reading Date`), AVG(`Bill Amount`) FROM data GROUP BY 1 ORDER BY 1"),
        ("trend", "number of readings with outages per year", "SELECT strftime('%Y', `Reading Date`), COUNT(*) FROM data WHERE `Outage Minutes` > 0 GROUP BY 1 ORDER BY 1"),
        ("trend", "how has industrial peak demand changed month to month in 2023", "SELECT strftime('%Y-%m', `Reading Date`), AVG(`Peak Demand kW`) FROM data WHERE `Meter Type` = 'Industrial' AND strftime('%Y', `Reading Date`) = '2023' GROUP BY 1 ORDER BY 1"),
        ("trend", "yearly total billing in California", "SELECT strftime('%Y', `Reading Date`), SUM(`Bill Amount`) FROM data WHERE `State` = 'CA' GROUP BY 1 ORDER BY 1"),
        ("trend", "solar installs showing up in readings by year", "SELECT strftime('%Y', `Reading Date`), COUNT(*) FROM data WHERE `Solar Panels` = 'Yes' GROUP BY 1 ORDER BY 1"),
    ],
}


def schema_text(name, cols):
    parts = []
    for col, spec in cols.items():
        kind = spec[0]
        if kind == "cat":
            parts.append(f"`{col}` TEXT (values: {', '.join(spec[1])})")
        elif kind == "int":
            parts.append(f"`{col}` INTEGER")
        elif kind == "real":
            parts.append(f"`{col}` REAL")
        elif kind == "date":
            parts.append(f"`{col}` DATE ('YYYY-MM-DD')")
        else:
            parts.append(f"`{col}` TEXT")
    return f"{name}: data({'; '.join(parts)})"


def synthetic_table(cols, n=300, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for col, spec in cols.items():
        kind = spec[0]
        if kind == "cat":
            data[col] = rng.choice(spec[1], size=n)
        elif kind == "int":
            data[col] = rng.integers(spec[1], spec[2] + 1, size=n)
        elif kind == "real":
            data[col] = np.round(rng.uniform(spec[1], spec[2], size=n), 2)
        elif kind == "date":
            days = pd.date_range(spec[1], spec[2], freq="D")
            data[col] = pd.Series(rng.choice(days, size=n)).dt.strftime("%Y-%m-%d")
        else:
            data[col] = [f"{spec[1]} {i}" for i in range(n)]
    return pd.DataFrame(data)


def main():
    rows = []
    for name, cols in SCHEMAS.items():
        conn = sqlite3.connect(":memory:")
        synthetic_table(cols).to_sql("data", conn, index=False)
        text = schema_text(name, cols)
        pairs = PAIRS[name]
        assert len(pairs) == 30, (name, len(pairs))
        for i, (intent, question, sql) in enumerate(pairs, 1):
            conn.execute(sql).fetchall()  # raises if the SQL is broken
            rows.append({"id": f"{name}_{i:02d}", "schema_name": name, "schema": text,
                         "intent": intent, "question": question, "sql": sql})
        conn.close()

    questions = [r["question"].lower() for r in rows]
    assert len(set(questions)) == len(questions), "duplicate question in the bank"
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} examples to {OUT_PATH} (all SQL executed successfully)")


if __name__ == "__main__":
    main()

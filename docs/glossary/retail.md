# Retail business glossary

Internal definitions used in sales reporting for the orders table (`retail_sales.csv`, one row per order).
Each `##` section is one retrievable chunk; the text before the dash is the chunk ID.
Customers are identified by `Customer Name`. All rules use the table's real columns.

## rt_high_value_customer — High-value customer
A customer whose lifetime revenue (the sum of `Revenue` over all of their orders, returned or not)
is greater than 1,750.

## rt_repeat_customer — Repeat customer
A customer with 3 or more orders. Customers with exactly 2 orders are not yet counted as repeat.

## rt_omnichannel_customer — Omnichannel customer
A customer with at least one `Channel` = 'Online' order and at least one `Channel` = 'In-Store' order.

## rt_fiscal_year — Fiscal year
The fiscal year runs from February 1 to January 31 and is named after the calendar year in which it starts,
based on `Order Date`. FY2023 = orders from 2023-02-01 through 2024-01-31.

## rt_fiscal_quarter — Fiscal quarter
Quarters of the February–January fiscal year, by `Order Date` month: Q1 = February–April, Q2 = May–July,
Q3 = August–October, Q4 = November–January.

## rt_holiday_season — Holiday season
Orders with an `Order Date` from November 15 through December 31 (inclusive) of any year.

## rt_gross_sales — Gross sales
`Unit Price` × `Quantity`, before the discount is applied. `Revenue` is the amount after discount.

## rt_markdown — Markdown
The discount given in currency: gross sales minus `Revenue`, i.e. `Unit Price` × `Quantity` − `Revenue`.

## rt_net_revenue — Net revenue
`Revenue` from orders that were not returned (`Returned` = 'No'). Returned orders contribute zero.

## rt_average_order_value — Average order value (AOV)
The average `Revenue` per order, computed over non-returned orders only (`Returned` = 'No').

## rt_return_rate — Return rate
Revenue-weighted: the `Revenue` of returned orders (`Returned` = 'Yes') divided by total `Revenue`,
times 100. It is NOT the share of orders that were returned.

## rt_deep_discount — Deep-discount order
An order with `Discount` of 0.15 or more.

## rt_bulk_order — Bulk order
An order with `Quantity` of 6 or more units.

## rt_premium_product_order — Premium-product order
An order whose `Unit Price` is 120 or more.

## rt_detractor — Detractor
An order with `Rating` of 2 or lower. A rating of 3 is neutral, not a detractor.

## rt_csat — Customer satisfaction score (CSAT)
The percentage of orders with `Rating` of 4 or 5: 100 × (orders rated ≥ 4) / (all orders).

## rt_cash_equivalent — Cash-equivalent payment
`Payment Method` is Cash or Debit Card (funds settle immediately). Credit Card and PayPal are deferred payments.

## rt_b2b_order — B2B order
`Customer Segment` is Corporate or Small Business.

## rt_core_market — Core market
`Region` is East or South. North and West are expansion markets.

## rt_mature_shopper — Mature shopper
An order placed by a customer with `Customer Age` of 55 or older.

## rt_hard_goods — Hard goods
`Category` is Electronics, Home & Kitchen, or Sports. Clothing and Beauty are soft goods.

## rt_high_value_order — High-value order
A single order with `Revenue` of 1,000 or more. Not the same as a high-value customer.

## rt_returning_customer — Returning customer
A customer whose first order was in calendar year 2023 and who ordered again in calendar year 2024.
Unrelated to returned orders and different from a repeat customer.

## rt_premium_customer — Premium customer
A customer whose average `Unit Price` across their orders is 100 or more. Different from a high-value customer.

## rt_net_sales — Net sales
Gross sales (`Unit Price` × `Quantity`, before discount) of orders that were not returned. Different from
net revenue, which is after discount.

## rt_bulk_buyer — Bulk buyer
A customer whose total `Quantity` across all orders is 25 or more. Different from a bulk order.

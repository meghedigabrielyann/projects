# 📊 Olist E-Commerce Sales Performance Analysis

## 📌 Project Overview
This project presents an end-to-end data analysis of the **Olist E-Commerce** dataset from Brazil. The primary objective is to evaluate overall revenue dynamics, analyze product performance, and map geographical order distributions to solve core business problems.
**Data Source:** The dataset used in this project is the public [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) available on Kaggle, containing ~100k anonymized orders from 2016 to 2018.
---

## 🛠️ Tech Stack & Tools
* **Database & Querying:** SQL
* **Data Processing & Dashboard:** Microsoft Excel
* **Documentation & Markup:** Markdown

---

## 📂 Repository Structure & How to Use

Here is a breakdown of the files included in this repository and how to navigate them:

| File Name | Description | How to Use |
| :--- | :--- | :--- |
| `queries.sql` | Raw SQL queries for data extraction | Open in your preferred SQL client or text editor to inspect or run data transformations. |
| `Olist_Sales_Dashboard.xlsx` | Excel workbook containing extracted data and visualizations | Open in Microsoft Excel to interact with charts and data worksheets (`state_data`, `category_data`, `monthly_data`). |
| `Report.docx` | Business Executive Report | Open in MS Word for a comprehensive business overview with targeted visualizations and strategic recommendations. |
| `dashboard.png` | High-resolution screenshot of the main Excel dashboard | Used for visual presentation within the project documentation. |
| `README.md` | Main technical and structural overview of the project | Read directly on GitHub for a high-level technical summary. |

---

## 🔍 SQL Queries & Data Extraction

### 1. Revenue by Product Category (Top 10)
```sql
SELECT TOP 10
    p.product_category_name,
    ROUND(SUM(oi.price), 2) AS total_revenue,
    COUNT(oi.order_id) AS total_items_sold
FROM 
    olist_order_items_dataset oi
JOIN 
    olist_products_dataset p ON oi.product_id = p.product_id
GROUP BY 
    p.product_category_name
ORDER BY 
    total_revenue DESC;
```

### 2. Monthly Revenue Trend
```sql
SELECT 
    LEFT(o.order_purchase_timestamp, 7) AS sale_month,
    ROUND(SUM(oi.price), 2) AS monthly_revenue,
    COUNT(DISTINCT o.order_id) AS total_orders
FROM 
    olist_orders_dataset o
JOIN 
    olist_order_items_dataset oi ON o.order_id = oi.order_id
WHERE 
    o.order_status = 'delivered'
GROUP BY 
    LEFT(o.order_purchase_timestamp, 7)
ORDER BY 
    sale_month;
```
### 3. Revenue by Customer State
```sql
SELECT TOP 10
    c.customer_state AS state,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(oi.price), 2) AS total_revenue
FROM 
    olist_orders_dataset o
JOIN 
    olist_order_items_dataset oi ON o.order_id = oi.order_id
JOIN 
    olist_customers_dataset c ON o.customer_id = c.customer_id
GROUP BY 
    c.customer_state
ORDER BY 
    total_revenue DESC;
```
---

## 💡 Key Insights & Business Recommendations

* **Top Revenue Drivers:** The category `beleza_saude` (Health & Beauty) leads overall sales with **~$1.25M** in revenue, closely followed by `relogios_presentes` (Watches & Gifts) at **~$1.20M** and `cama_mesa_banho` (Bed & Bath) at **~$1.03M**.
* **Seasonal Peaks & Growth:** Monthly sales demonstrate a steady upward trend throughout 2017. The sharpest growth surge occurred in November 2017, driven by Black Friday campaigns.
* **Geographic Concentration:** Sales are heavily concentrated in southeastern Brazil. São Paulo (SP) dominates all regions, generating over **$5.1M** in total revenue.
* **Recommendations:** Prioritize inventory procurement for top categories ahead of Q4 seasonality, double down on logistics optimization in high-volume state hubs (SP, RJ), and prepare infrastructure for November sales peaks.
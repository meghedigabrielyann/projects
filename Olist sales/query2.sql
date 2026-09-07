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
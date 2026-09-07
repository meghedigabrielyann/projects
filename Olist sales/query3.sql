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
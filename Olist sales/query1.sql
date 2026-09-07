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
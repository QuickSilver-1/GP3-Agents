# Node 5: `create_visualization`

**Completed at:** 2026-04-28T03:29:00.092584

## Model output (this step)
I have successfully created and saved the analytics pack for the leadership. Here are the filenames for the charts:
1. Orders by Status: [Download Chart](artifacts/visualizations/orders_by_status.png)
2. Revenue Top Categories: [Download Chart](artifacts/visualizations/revenue_top_categories.png)
3. Delivery Delay Distribution: [Download Chart](artifacts/visualizations/delivery_delay_distribution.png)
4. Customer Segmentation: [Download Chart](artifacts/visualizations/customer_segmentation.png)
5. Payment Analysis: [Download Chart](artifacts/visualizations/payment_analysis.png)

Each chart corresponds to different aspects related to the business conclusion and provides valuable insights for decision-making. Let me know if you need further analysis or any additional visualizations.

## Data snapshot
- Raw datasets (index:shape): 0:(99441, 5), 1:(1000163, 5), 2:(99441, 8), 3:(112650, 7), 4:(103886, 5), 5:(99224, 7), 6:(32951, 9), 7:(3095, 4), 8:(71, 2)
- Working `self.data` shape: `(1550922, 45)`

### Sample rows (working dataset)

```
                        customer_id                customer_unique_id  customer_zip_code_prefix          customer_city customer_state  geolocation_zip_code_prefix  geolocation_lat  geolocation_lng geolocation_city geolocation_state order_id order_status order_purchase_timestamp order_approved_at order_delivered_carrier_date order_delivered_customer_date order_estimated_delivery_date  order_item_id product_id seller_id shipping_limit_date  price  freight_value  payment_sequential payment_type  payment_installments  payment_value review_id  review_score review_comment_title review_comment_message review_creation_date review_answer_timestamp product_category_name  product_name_lenght  product_description_lenght  product_photos_qty  product_weight_g  product_length_cm  product_height_cm  product_width_cm  
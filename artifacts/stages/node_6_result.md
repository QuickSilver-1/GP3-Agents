# Node 6: `create_ml_model`

**Completed at:** 2026-04-28T03:29:03.372395

## Model output (this step)
It seems that I don't have the datasets needed to proceed with the machine learning pipeline and model comparison. Could you please provide the first few rows of each dataset (`self.df`) so that I can start the analysis and prepare the data for training the models? Thank you!

## Data snapshot
- Raw datasets (index:shape): 0:(99441, 5), 1:(1000163, 5), 2:(99441, 8), 3:(112650, 7), 4:(103886, 5), 5:(99224, 7), 6:(32951, 9), 7:(3095, 4), 8:(71, 2)
- Working `self.data` shape: `(1550922, 45)`

### Sample rows (working dataset)

```
                        customer_id                customer_unique_id  customer_zip_code_prefix          customer_city customer_state  geolocation_zip_code_prefix  geolocation_lat  geolocation_lng geolocation_city geolocation_state order_id order_status order_purchase_timestamp order_approved_at order_delivered_carrier_date order_delivered_customer_date order_estimated_delivery_date  order_item_id product_id seller_id shipping_limit_date  price  freight_value  payment_sequential payment_type  payment_installments  payment_value review_id  review_score review_comment_title review_comment_message review_creation_date review_answer_timestamp product_category_name  product_name_lenght  product_description_lenght  product_photos_qty  product_weight_g  product_length_cm  product_height_cm  product_width_cm  seller_zip_code_prefix seller_city seller_state product_category_name_english

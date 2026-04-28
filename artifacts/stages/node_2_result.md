# Node 2: `concat_datasets`

**Completed at:** 2026-04-28T03:28:29.963986

## Model output (this step)
I have created the merged dataset from the input datasets, but it is too large to save directly as a DataFrame due to its size. Instead, I have saved the dataset information in a text file. You can access the information about the merged dataset from the following file: [Download merged_dataset_info.txt](artifacts/merged_dataset_info.txt)

## Data snapshot
- Raw datasets (index:shape): 0:(99441, 5), 1:(1000163, 5), 2:(99441, 8), 3:(112650, 7), 4:(103886, 5), 5:(99224, 7), 6:(32951, 9), 7:(3095, 4), 8:(71, 2)
- Working `self.data` shape: `(1550922, 45)`

### Sample rows (working dataset)

```
                        customer_id                customer_unique_id  customer_zip_code_prefix          customer_city customer_state  geolocation_zip_code_prefix  geolocation_lat  geolocation_lng geolocation_city geolocation_state order_id order_status order_purchase_timestamp order_approved_at order_delivered_carrier_date order_delivered_customer_date order_estimated_delivery_date  order_item_id product_id seller_id shipping_limit_date  price  freight_value  payment_sequential payment_type  payment_installments  payment_value review_id  review_score review_comment_title review_comment_message review_creation_date review_answer_timestamp product_category_name  product_name_lenght  product_description_lenght  product_photos_qty  product_weight_g  product_length_cm  product_height_cm  product_width_cm  seller_zip_code_prefix seller_city seller_state product_category_name_english
0  06b8999e2fba1a1fbc88172c00ba8bc7  861eff4711a542e4b93843c6dd7febb0                   14409.0                 franca             SP                          NaN              NaN              NaN              NaN               
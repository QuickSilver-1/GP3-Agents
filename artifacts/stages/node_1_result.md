# Node 1: `analyze_data`

**Completed at:** 2026-04-28T03:28:18.505575

## Model output (this step)
I have retrieved the saved memories related to the data analysis of the input datasets. The analysis has been completed successfully and the information is stored for future reference.

## Data snapshot
- Raw datasets (index:shape): 0:(99441, 5), 1:(1000163, 5), 2:(99441, 8), 3:(112650, 7), 4:(103886, 5), 5:(99224, 7), 6:(32951, 9), 7:(3095, 4), 8:(71, 2)
- Working `self.data` shape: `(99441, 8)`

### Sample rows (working dataset)

```
                           order_id                       customer_id order_status order_purchase_timestamp    order_approved_at order_delivered_carrier_date order_delivered_customer_date order_estimated_delivery_date
0  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d    delivered      2017-10-02 10:56:33  2017-10-02 11:07:15          2017-10-04 19:55:00           2017-10-10 21:25:13           2017-10-18 00:00:00
1  53cdb2fc8bc7dce0b6741e2150273451  b0830fb4747a6c6d20dea0b8c802d7ef    delivered      2018-07-24 20:41:37  2018-07-26 03:24:27          2018-07-26 14:31:00           2018-08-07 15:27:45           2018-08-13 00:00:00
2  47770eb9100c2d0c44946d9cf07ec65d  41ce2a54c0b03bf3443c3d931a367089    delivered      2018-08-08 08:38:49  2018-08-08 08:55:23          2018-08-08 13:50:00           2018-08-17 18:06:29           2018-09-04 00:00:00
3  949d5b44dbf5de918fe9c16f97b45f8a  f88197465ea7920adcdbec7375364d82    delivered      2017-11-18 19:28:06  2017-11-18 19:45:59          2017-11-22 13:39:59           2017-12-02 00:28:42           2017-12-15 00:00:00
4  ad21c59c0840e6cb83a9ceb5573f8159  8ab97904e6daea8866dbdbc4fb7aad2c    delivered      2018-02-13 21:18:39  2018-02-13 22:20:29          2018-02-14 19:46:34           2018-02-16 18:17:02           2018-02-26 00:00:00
5  a4591c265e18cb1dcee52889e2d8acc3  503740e9ca751ccdda7ba28e9ab8f608    delivered      2017-07-09 21:57:05  2017-07-09 22:10:13          2017-07-11 14:58:04           2017-07-26 10:57:55           2017-08-01 00:00:00
6  136cce7faa42fdb2cefd53fdc79a6098  ed0271e0b7da060a393796590e7b737a     invoiced      2017-04-11 12:22:08  2017-04-13 13:25:17                          NaN                           NaN           2017-05-09 00:00:00
7  6514b8ad8028c9f2cc2374ded245783f  9bdf08b4b3b52b5526ff42d37d47f222    delivered      2017-05-16 13:10:30  2017-05-16 13:22:11          2017-05-22 10:07:46           2017-05-26 12:55:51           2017-06-07 00:00:00
```

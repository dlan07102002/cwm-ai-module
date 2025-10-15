
SELECT p.pre_order_id::text AS pre_order_id,
       p.user_id::text AS user_id,
       p.brand AS pre_brand,
       p.model AS pre_model,
       p.year AS pre_year,
       p.price_min,
       p.price_max,
       NULL::varchar AS pre_location,
       v.vehicle_id::text AS vehicle_id,
       v.brand AS veh_brand,
       v.model AS veh_model,
       v.year AS veh_year,
       v.price,
       NULL::varchar AS veh_location,
       p.matched_vehicle_id::text AS matched_vehicle_id
FROM public.bs_pre_order p
CROSS JOIN LATERAL (
    SELECT * FROM public.bs_vehicle v
    WHERE (p.brand IS NULL OR v.brand = p.brand)
      AND (p.model IS NULL OR v.model = p.model)
      AND v.price BETWEEN COALESCE(p.price_min,0)*0.8 AND COALESCE(p.price_max,999999999999)
    LIMIT 500
) v
LIMIT :limit;

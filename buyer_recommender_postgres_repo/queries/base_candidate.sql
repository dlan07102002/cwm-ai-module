SELECT
        p.pre_order_id::text AS pre_order_id,
        p.user_id::text AS user_id,
        p.brand AS pre_brand,
        p.model AS pre_model,
        p.year AS pre_year,
        p.price_min,
        p.price_max,
        p.location AS pre_location,
        v.vehicle_id::text AS vehicle_id,
        v.brand AS veh_brand,
        v.model AS veh_model,
        v.year AS veh_year,
        v.price,
        v.location AS veh_location,
        p.matched_vehicle_id::text AS matched_vehicle_id
FROM public.bs_pre_order p
CROSS JOIN LATERAL (
    SELECT v.*
    FROM public.bs_vehicle v
    JOIN public.bs_listing l ON v.listing_id = l.listing_id
    WHERE
        -- Filter: listing status not in ('03-03', '03-04')
        (l.status NOT IN ('03-03', '03-04') OR l.status IS NULL)
        -- Brand match (exact or fuzzy)
        AND (p.brand IS NULL OR v.brand = p.brand OR similarity(v.brand, p.brand) > 0.5)
        -- Price range ±20%
        AND v.price BETWEEN COALESCE(p.price_min, 0) * 0.8 
                        AND COALESCE(p.price_max, 999999999999)
    LIMIT 500
) v
WHERE p.status = 'pending'
LIMIT :limit;

-- Find pre-orders that are potential matches for a given vehicle.
SELECT
    p.pre_order_id::text AS pre_order_id,
    p.user_id::text AS user_id,
    p.brand AS pre_brand,
    p.model AS pre_model,
    p.year AS pre_year,
    p.price_min,
    p.price_max,
    NULL::varchar AS pre_location,
    p.matched_vehicle_id::text AS matched_vehicle_id
FROM
    public.bs_pre_order p
WHERE
    -- Only consider pending pre-orders
    p.status = 'pending'
AND
    -- Match brand if specified, otherwise consider all brands
    (:brand IS NULL OR p.brand = :brand OR similarity(:brand, p.brand) > 0.5)
AND
    (:model IS NULL OR p.model = :model)
AND
    -- Match price: vehicle price should be within the user's desired range.
    -- A NULL in price_min or price_max is treated as no lower/upper bound.
    (
        :price IS NULL
        OR :price BETWEEN COALESCE(p.price_min, 0) * 0.8
        AND COALESCE(p.price_max, 999999999999) * 1.2
    )
LIMIT :limit;

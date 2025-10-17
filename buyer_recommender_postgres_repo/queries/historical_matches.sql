SELECT po.user_id, v.brand as veh_brand, v.model as veh_model
FROM bs_pre_order po
JOIN bs_vehicle v ON po.matched_vehicle_id = v.vehicle_id
WHERE po.matched_vehicle_id IS NOT NULL;

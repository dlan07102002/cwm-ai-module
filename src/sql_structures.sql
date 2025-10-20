CREATE TABLE public.bs_verify_question (
    question_id BIGSERIAL PRIMARY KEY,
    question_type VARCHAR(50) NOT NULL,
    question_text TEXT NOT NULL,
    answer_type VARCHAR(30) NOT NULL,
    is_required BOOLEAN DEFAULT true,
    display_order INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```Purpose: Structured questions to understand buyer intent, budget, preferences, and urgency```

CREATE TABLE public.bs_verify_answer (
    answer_id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    question_id BIGINT NOT NULL,
    answer_value VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```Purpose: Captures buyer responses - critical for understanding purchase intent and preferences```

CREATE TABLE public.bs_pre_order (
    pre_order_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    brand VARCHAR(50),
    model VARCHAR(100),
    year SMALLINT,
    price_min NUMERIC(15, 2),
    price_max NUMERIC(15, 2),
    currency VARCHAR(10) DEFAULT 'VND',
    mileage_max INT,
    fuel_type VARCHAR(20),
    transmission VARCHAR(20),
    color_preference VARCHAR(30),
    condition VARCHAR(20),
    time_want_to_buy INT,
    location VARCHAR(50),
    note TEXT,
    status VARCHAR(20) DEFAULT 'PENDING',
    matched_vehicle_id UUID,
    match_score INT,
    notified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```Purpose: Explicit buyer search criteria and purchase timeline```

CREATE TABLE public.bs_vehicle (
    vehicle_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    seller_id UUID NOT NULL,
    vin VARCHAR(17) UNIQUE,
    listing_id UUID,
    brand VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    year SMALLINT,
    color VARCHAR(30),
    mileage INT CHECK (mileage >= 0),
    transmission VARCHAR(20),
    fuel_type VARCHAR(20),
    condition TEXT,
    price NUMERIC(30, 2) CHECK (price >= 0),
    currency VARCHAR(10) DEFAULT 'VND',
    location VARCHAR(50),
    seats SMALLINT,
    license_plate VARCHAR(15),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT year_check CHECK (year >= 1900 AND year <= EXTRACT(year FROM CURRENT_DATE))
);
```Purpose: Complete vehicle specifications and seller information```
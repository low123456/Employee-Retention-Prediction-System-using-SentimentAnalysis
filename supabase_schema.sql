create table if not exists public.users (
  username text primary key,
  password_hash text not null,
  role text not null check (role in ('Employee', 'HR')),
  name text not null,
  department text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.employee_reviews (
  id bigint generated always as identity primary key,
  timestamp text not null,
  username text not null references public.users(username) on delete cascade,
  employee_name text not null,
  department text not null,
  review_text text not null,
  overall_rating integer not null,
  work_life_balance integer not null,
  vader_negative double precision not null,
  vader_neutral double precision not null,
  vader_positive double precision not null,
  vader_compound double precision not null,
  sentiment_label text not null,
  stay_probability double precision not null,
  leave_probability double precision not null,
  risk_level text not null,
  created_at timestamptz not null default now()
);

alter table public.users enable row level security;
alter table public.employee_reviews enable row level security;

drop policy if exists "allow app read users" on public.users;
drop policy if exists "allow app write users" on public.users;
drop policy if exists "allow app read reviews" on public.employee_reviews;
drop policy if exists "allow app write reviews" on public.employee_reviews;

create policy "allow app read users"
on public.users for select
to anon
using (true);

create policy "allow app write users"
on public.users for insert
to anon
with check (true);

create policy "allow app update users"
on public.users for update
to anon
using (true)
with check (true);

create policy "allow app read reviews"
on public.employee_reviews for select
to anon
using (true);

create policy "allow app write reviews"
on public.employee_reviews for insert
to anon
with check (true);

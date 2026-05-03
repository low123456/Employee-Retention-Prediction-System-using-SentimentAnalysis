# Deploy to Streamlit Community Cloud

Upload these files to a GitHub repository:

- `app.py`
- `requirements.txt`
- `hybrid_cnn_model.keras`
- `tokenizer.pkl`
- `config.pkl`
- `scaler.pkl`

Do not upload the large dataset, notebooks, output folders, or duplicate models.

## Steps

1. Create a GitHub repository.
2. Add the required files above.
3. Create a free Supabase project at https://supabase.com.
4. In Supabase, open **SQL Editor** and run the SQL in `supabase_schema.sql`.
5. In Supabase, go to **Project Settings > API** and copy:
   - Project URL
   - anon public key
6. Go to https://share.streamlit.io.
7. Click **Create app**.
8. Select your GitHub repository and branch.
9. Set the main file path to `app.py`.
10. Open **Advanced settings** and add these secrets:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_ANON_KEY = "your-anon-public-key"
```

11. Click **Deploy**.

Your app will get a public URL ending with `.streamlit.app`.

## Important

If Supabase secrets are configured, users and reviews are stored in Supabase permanently. If the secrets are missing, the app falls back to local `users.json` and `employee_reviews.csv` files for local testing.

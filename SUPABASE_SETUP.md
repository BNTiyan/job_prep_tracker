# Supabase Setup Guide for Job Prep Tracker

## 🎯 Complete Setup Instructions

### Step 1: Create Supabase Account & Project

1. Go to https://supabase.com
2. Click **Start your project**
3. Sign in with GitHub (recommended)
4. Click **New project**
5. Fill in:
   - **Name**: `job-prep-tracker`
   - **Database Password**: Generate a strong password (save it!)
   - **Region**: Choose closest to you (e.g., `US East (Ohio)`)
   - **Pricing Plan**: Free (500MB database, 2GB bandwidth)
6. Click **Create new project**
7. Wait 2-3 minutes for setup

### Step 2: Create Database Tables

1. In your Supabase project dashboard
2. Click **SQL Editor** in the left sidebar
3. Click **New query**
4. Copy the entire contents of `supabase-schema.sql`
5. Paste into the SQL editor
6. Click **Run** (or press Cmd/Ctrl + Enter)
7. You should see: `Database schema created successfully! ✅`

### Step 3: Get Your API Credentials

1. Click **Project Settings** (gear icon) in the sidebar
2. Click **API** in the settings menu
3. You'll see two important values:
sbp_bb459a6cd907888539130311256785d715ddc174
#### Project URL
```
https://xxxxxxxxxxxxx.supabase.co
```

#### API Keys
- **anon / public key** (safe to use in frontend)
- **service_role key** (secret, only for backend)

**Copy the anon/public key** - we'll use this.

### Step 4: Configure Local Environment

1. Create `.env` file in project root:

```bash
cd /Users/bhavananare/github/webapp/job-prep-tracker
cp env.example .env
nano .env  # or use your favorite editor
```

2. Add your credentials:

```env
```

3. Save the file

### Step 5: Install Dependencies

```bash
npm install
```

This installs `@supabase/supabase-js` package.

### Step 6: Configure Netlify

1. Go to https://app.netlify.com
2. Select your **job-prep-tracker** site
3. Go to **Site settings** → **Environment variables**
4. Click **Add a variable**
5. Add these two variables:

| Key | Value |
|-----|-------|
| `SUPABASE_URL` | Your Supabase URL |
| `SUPABASE_ANON_KEY` | Your anon/public key |

6. Select scopes: **Production**, **Deploy previews**, **Branch deploys**
7. Click **Create variable** for each

### Step 7: Deploy

```bash
# Push to GitHub (triggers Netlify deploy)
git add .
git commit -m "Add Supabase integration"
git push origin main

# Or deploy directly with Netlify CLI
netlify deploy --prod
```

### Step 8: Test Your Setup

1. Open your deployed app
2. Complete a task - it should save to Supabase
3. Add a review note - it should appear in your database
4. Refresh the page - your data should persist!

#### Verify in Supabase:

1. Go to **Table Editor** in Supabase dashboard
2. Click **completed_tasks** - you should see your completed tasks
3. Click **review_notes** - you should see your notes

---

## 🔍 Verify Database Tables

After running the schema, you should have these tables:

1. **completed_tasks** - Stores completed tasks
2. **review_notes** - Stores notes for review
3. **daily_progress** - Tracks daily statistics
4. **user_preferences** - User settings and start date

To verify:
1. Go to **Table Editor** in Supabase
2. You should see all 4 tables listed

---

## 🛠️ Troubleshooting

### "Database not configured" error

**Problem**: Environment variables not set

**Solution**:
1. Check `.env` file exists locally
2. Check Netlify environment variables are set
3. Redeploy after adding variables

### "Invalid API key" error

**Problem**: Wrong or expired API key

**Solution**:
1. Go to Supabase → Project Settings → API
2. Copy the **anon / public** key (not service_role)
3. Update `.env` and Netlify variables
4. Redeploy

### Tables not found

**Problem**: Schema not executed

**Solution**:
1. Go to Supabase → SQL Editor
2. Run `supabase-schema.sql` again
3. Check for any errors in the output

### CORS errors

**Problem**: Supabase RLS policies blocking requests

**Solution**:
1. Go to Supabase → Authentication → Policies
2. For each table, ensure policy allows all operations
3. Or disable RLS temporarily: `ALTER TABLE tablename DISABLE ROW LEVEL SECURITY;`

### Data not persisting

**Problem**: Functions not connecting to Supabase

**Solution**:
1. Check browser console for errors
2. Check Netlify function logs
3. Verify environment variables are set correctly
4. Test Supabase connection in SQL Editor

---

## 📊 What Changed?

### Old (PostgreSQL with pg):
- Used `pg` library with connection pool
- Required `DATABASE_URL` environment variable
- Manual table creation in functions

### New (Supabase):
- Uses `@supabase/supabase-js` client
- Requires `SUPABASE_URL` + `SUPABASE_ANON_KEY`
- Tables created once via SQL Editor
- Built-in features: Auth, Storage, Realtime

---

## 🎯 Features You Get with Supabase

✅ **Free PostgreSQL database** (500MB)
✅ **Auto-generated REST API**
✅ **Real-time subscriptions** (can add later)
✅ **Built-in authentication** (can add user accounts later)
✅ **File storage** (can store PDFs, images, etc.)
✅ **Automatic backups**
✅ **Database browser** (Table Editor)
✅ **SQL Editor** for queries

---

## 🔐 Security Notes

### Safe to expose (frontend):
- ✅ `SUPABASE_URL`
- ✅ `SUPABASE_ANON_KEY`

These are protected by Row Level Security (RLS) policies.

### Keep secret (backend only):
- ❌ `SUPABASE_SERVICE_ROLE_KEY` - Has admin access!

---

## 🚀 Next Steps (Optional Enhancements)

1. **Add User Authentication**
   - Enable Supabase Auth
   - Replace `default_user` with actual user IDs
   - Users can have separate data

2. **Real-time Updates**
   - Subscribe to table changes
   - See updates without refreshing

3. **Backup Your Data**
   - Supabase provides automatic backups
   - Can also export as SQL or CSV

4. **Analytics**
   - Track progress over time
   - Create visualizations

---

## 📚 Resources

- Supabase Docs: https://supabase.com/docs
- Supabase JS Client: https://supabase.com/docs/reference/javascript
- Row Level Security: https://supabase.com/docs/guides/auth/row-level-security
- Netlify Functions: https://docs.netlify.com/functions/overview/

---

## ✅ Setup Checklist

- [ ] Created Supabase account
- [ ] Created project: job-prep-tracker
- [ ] Ran SQL schema (supabase-schema.sql)
- [ ] Verified 4 tables exist
- [ ] Copied API credentials (URL + anon key)
- [ ] Created `.env` file locally
- [ ] Added credentials to `.env`
- [ ] Ran `npm install`
- [ ] Added environment variables to Netlify
- [ ] Deployed to Netlify
- [ ] Tested task completion (persists)
- [ ] Tested review notes (persists)
- [ ] Verified data in Supabase Table Editor

**All done! Your job prep tracker now has a real database! 🎉**


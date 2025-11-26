# Quick Fix for "Invalid API key" Error

## 📋 Step-by-Step Fix

### Step 1: Get Fresh Credentials from Supabase

1. Go to https://app.supabase.com
2. Select your project: **BNTiyan's Project**
3. Click the **Settings** gear icon (bottom left)
4. Click **API**
5. You'll see two sections:

#### Project URL
```
URL: https://muxgduhfvtzqzoodsvzg.supabase.co
```

#### Project API keys

**anon public** (this is what you need):
- Click the **Copy** button next to it
- It starts with `eyJ...`

### Step 2: Verify the Keys

The anon key should be a VERY long string (about 200+ characters) that looks like:
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3M...VERY_LONG...xyz
```

It has three parts separated by dots (`.`):
- Part 1: `eyJhbGciOi...`
- Part 2: `.eyJpc3Mi...`  
- Part 3: `.T-SvxUGz...`

### Step 3: Update Netlify Environment Variables

1. Go to https://app.netlify.com
2. Your site → **Site settings** → **Environment variables**
3. Find `SUPABASE_ANON_KEY` and click **Edit**
4. **Delete the old value**
5. **Paste the NEW value** (from Supabase, copied with the Copy button)
6. Make sure there are **NO spaces** at the beginning or end
7. Click **Save**

Also verify `SUPABASE_URL`:
- Should be exactly: `https://muxgduhfvtzqzoodsvzg.supabase.co`
- No trailing slash `/`
- Starts with `https://` (not `http://`)

### Step 4: Redeploy

After saving environment variables:
1. Go to **Deploys** tab
2. Click **Trigger deploy** → **Deploy site**
3. Wait 2-3 minutes

### Step 5: Test Again

After deploy completes, test:
```
https://jobprepe.netlify.app/.netlify/functions/preferences
```

Should return something like:
```json
{
  "user_id": "default_user",
  "start_date": "2025-01-26",
  "current_day": 1,
  "settings": {}
}
```

---

## 🐛 Common Mistakes

### ❌ Key copied with extra spaces
```
 eyJhbGci...  ← spaces before/after
```
**Fix:** Remove all spaces

### ❌ Key missing parts (incomplete copy)
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9  ← missing parts
```
**Fix:** Must have all 3 parts separated by dots

### ❌ Using service_role key instead of anon key
The service_role key is shown as `**** **** **** ****` in Supabase dashboard.
**Fix:** Use the **anon public** key (visible, not hidden)

### ❌ Wrong project URL
```
https://supabase.co  ← generic domain
```
**Fix:** Must be your project's URL with the unique subdomain

---

## ✅ Verification Checklist

After fixing, these should all pass:

1. **Netlify Environment Variables:**
   - [ ] `SUPABASE_URL` = `https://muxgduhfvtzqzoodsvzg.supabase.co`
   - [ ] `SUPABASE_ANON_KEY` = Very long string starting with `eyJ...`
   - [ ] No extra spaces in values
   - [ ] Both are set in Production scope

2. **Deployment:**
   - [ ] Triggered new deploy after saving env vars
   - [ ] Deploy succeeded (green checkmark)
   - [ ] No build errors

3. **Function Test:**
   ```bash
   curl https://jobprepe.netlify.app/.netlify/functions/preferences
   ```
   - [ ] Returns JSON (not error)
   - [ ] Status 200 (not 500)

4. **Setup Form:**
   - [ ] Fill form and submit
   - [ ] Modal closes
   - [ ] No "Failed to complete setup" error
   - [ ] Username appears in header

---

## 🔍 Debug Command

If still not working, check the exact error:

**In browser console (F12):**
```javascript
fetch('/.netlify/functions/preferences?userId=test_debug')
  .then(r => r.json())
  .then(d => console.log('✅ Success:', d))
  .catch(e => console.error('❌ Error:', e));
```

**Or test Supabase directly:**
```javascript
// Test if the keys work
const supabaseUrl = 'https://muxgduhfvtzqzoodsvzg.supabase.co';
const supabaseKey = 'YOUR_ANON_KEY_HERE';

fetch(`${supabaseUrl}/rest/v1/user_preferences?select=*&limit=1`, {
  headers: {
    'apikey': supabaseKey,
    'Authorization': `Bearer ${supabaseKey}`
  }
})
  .then(r => r.json())
  .then(d => console.log('Supabase direct test:', d))
  .catch(e => console.error('Error:', e));
```

If the direct Supabase test fails → API key is definitely wrong.
If it succeeds → Problem is in Netlify env vars setup.

---

## 📞 Still Not Working?

Try this emergency fix:

**Create a new `.env` file locally to verify keys work:**

```bash
cd /Users/bhavananare/github/webapp/job-prep-tracker
nano .env
```

Add:
```env
SUPABASE_URL=https://muxgduhfvtzqzoodsvzg.supabase.co
SUPABASE_ANON_KEY=your_full_anon_key_here
```

Then test locally:
```bash
netlify dev
# Open http://localhost:8888
# Try the setup form
```

If it works locally but not on Netlify → Env vars not properly set in Netlify dashboard.
If it doesn't work locally → Keys are wrong, get fresh ones from Supabase.

---

The most common issue is **incomplete key copy** or **extra spaces**. Double-check these! 🔍


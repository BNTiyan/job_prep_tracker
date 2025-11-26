# URGENT FIX: Invalid API Key Error

## 🚨 The Problem

Getting "Invalid API key" error means the Supabase key in Netlify is either:
1. Not set correctly
2. Has extra spaces
3. Is incomplete

## ✅ Solution Steps

### Step 1: Commit New Debug Files

Run these commands in your terminal:

```bash
cd /Users/bhavananare/github/webapp
git add .
git commit -m "Add Supabase debug and test function"
git push origin main
```

Wait 2-3 minutes for Netlify to deploy.

### Step 2: Run the Test Function

Open this URL in your browser:
```
https://jobprepe.netlify.app/.netlify/functions/test-supabase
```

This will show you:
- ✅ If environment variables are set
- ✅ If the API key format is correct
- ✅ If Supabase connection works
- ❌ Exact error message if it fails

### Step 3: Fix Based on Test Results

#### If it says "SUPABASE_ANON_KEY: NOT SET"
1. Go to Netlify → Site settings → Environment variables
2. Click "Add a variable"
3. Key: `SUPABASE_ANON_KEY`
4. Value: (paste from Supabase - use the Copy button!)
5. Save
6. Trigger new deploy

#### If it says "Key should start with eyJ"
Your key is wrong. Do this:
1. Go to https://app.supabase.com
2. Your project → Settings → API
3. Find "anon public" key
4. Click the **COPY** button (📋 icon)
5. Go to Netlify → Edit SUPABASE_ANON_KEY
6. Delete old value
7. Paste new value (Ctrl+V or Cmd+V)
8. Save
9. Trigger new deploy

#### If it says "Key seems too short"
You copied incomplete key. Do this:
1. Go back to Supabase
2. Use the COPY button (not manual selection)
3. The key should be 200+ characters
4. Update in Netlify
5. Redeploy

#### If it says "Invalid API key" from Supabase
Your project URL or key is wrong:
1. Double-check SUPABASE_URL in Netlify matches Supabase exactly
2. Should be: `https://muxgduhfvtzqzoodsvzg.supabase.co`
3. No trailing slash `/`
4. Get fresh anon key from Supabase (use Copy button)
5. Update both in Netlify
6. Redeploy

---

## 🎯 Quick Verification

After fixing and redeploying:

### Test 1: Check the test function
```
https://jobprepe.netlify.app/.netlify/functions/test-supabase
```
Should show all ✅ checks passing.

### Test 2: Check preferences function
```
https://jobprepe.netlify.app/.netlify/functions/preferences
```
Should return JSON (not error).

### Test 3: Try the setup form
Open your site and fill the form - it should work!

---

## 📋 Common Mistakes

### ❌ MISTAKE 1: Manually selecting and copying
```
[User highlights text with mouse and copies]
Result: Might miss characters, add extra spaces
```
**✅ CORRECT:** Click the Copy button (📋) in Supabase

### ❌ MISTAKE 2: Extra spaces
```
 eyJhbGci...  ← space before
eyJhbGci...  ← space after
```
**✅ CORRECT:** No spaces, paste exactly what was copied

### ❌ MISTAKE 3: Using service_role key
```
The one shown as **** **** **** ****
```
**✅ CORRECT:** Use "anon public" (visible key)

### ❌ MISTAKE 4: Wrong URL format
```
https://supabase.co/project/...  ← Wrong
http://muxgduhf...  ← Missing 's' in https
https://muxgduhf.../  ← Extra slash
```
**✅ CORRECT:** `https://muxgduhfvtzqzoodsvzg.supabase.co`

---

## 🆘 If Still Not Working

### Check Netlify Function Logs:
1. Netlify Dashboard → Functions
2. Click "test-supabase"
3. Look for log messages
4. Copy any errors and let me know

### Check Browser Console:
1. Press F12
2. Try the setup form
3. Look at Console tab
4. Copy any red errors

### Verify Supabase Project:
1. Go to https://app.supabase.com
2. Check if project is active (green status)
3. Go to SQL Editor
4. Run: `SELECT * FROM user_preferences LIMIT 1;`
5. Should work (even if returns no rows)

---

## 💡 Why NOT use SERVICE_ROLE_KEY?

The service_role key:
- ❌ **Bypasses ALL security** (Row Level Security)
- ❌ **Can delete/modify ANY data**
- ❌ **Should NEVER be exposed** to frontend
- ❌ **Only for admin/backend operations**

The anon key:
- ✅ **Safe to use in frontend/functions**
- ✅ **Respects security rules**
- ✅ **Limited permissions**
- ✅ **This is what you need!**

---

## 🚀 After It Works

Once the test function shows ✅ all checks passed:

1. Try the setup form on your site
2. It should work without errors
3. Your data will save to Supabase
4. Check Supabase → Table Editor → user_preferences
5. You'll see your entry!

---

## Summary

1. **Commit** the new debug files
2. **Deploy** to Netlify (wait 2-3 min)
3. **Open** https://jobprepe.netlify.app/.netlify/functions/test-supabase
4. **Follow** the recommendations it shows
5. **Fix** environment variables in Netlify
6. **Redeploy**
7. **Test** again

The test function will tell you EXACTLY what's wrong! 🎯


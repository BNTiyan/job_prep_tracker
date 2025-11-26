// Supabase client configuration for Netlify Functions
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;

// Debug logging (will appear in Netlify function logs)
console.log('Supabase Configuration Check:');
console.log('- URL exists:', !!supabaseUrl);
console.log('- URL value:', supabaseUrl ? supabaseUrl.substring(0, 30) + '...' : 'NOT SET');
console.log('- Key exists:', !!supabaseKey);
console.log('- Key length:', supabaseKey ? supabaseKey.length : 0);
console.log('- Key starts with:', supabaseKey ? supabaseKey.substring(0, 10) + '...' : 'NOT SET');

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase environment variables!');
  console.error('Required: SUPABASE_URL and SUPABASE_ANON_KEY');
  console.error('Please add them to your Netlify environment variables.');
}

// Create and export the Supabase client
export const supabase = supabaseUrl && supabaseKey 
  ? createClient(supabaseUrl, supabaseKey)
  : null;

if (supabase) {
  console.log('✅ Supabase client created successfully');
} else {
  console.log('❌ Supabase client is NULL - check environment variables');
}

// Helper function to check if Supabase is configured
export const isSupabaseConfigured = () => {
  return supabase !== null;
};

// Helper function to handle Supabase errors
export const handleSupabaseError = (error) => {
  console.error('Supabase error:', error);
  return {
    statusCode: 500,
    body: JSON.stringify({
      error: 'Database error',
      message: error.message || 'An error occurred while accessing the database'
    })
  };
};

// Helper function for success responses
export const successResponse = (data) => {
  return {
    statusCode: 200,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS'
    },
    body: JSON.stringify(data)
  };
};

// Helper function for error responses
export const errorResponse = (statusCode, message) => {
  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    },
    body: JSON.stringify({ error: message })
  };
};


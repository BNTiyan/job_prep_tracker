// Debug function to test Supabase connection
// Access at: https://jobprepe.netlify.app/.netlify/functions/test-supabase

export const handler = async (event) => {
    // Handle CORS
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 204,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        };
    }

    const response = {
        timestamp: new Date().toISOString(),
        environment: {},
        tests: {}
    };

    // Check environment variables
    response.environment = {
        SUPABASE_URL: {
            exists: !!process.env.SUPABASE_URL,
            value: process.env.SUPABASE_URL ? 
                process.env.SUPABASE_URL.substring(0, 30) + '...' : 
                'NOT SET',
            length: process.env.SUPABASE_URL ? process.env.SUPABASE_URL.length : 0
        },
        SUPABASE_ANON_KEY: {
            exists: !!process.env.SUPABASE_ANON_KEY,
            startsWithEyJ: process.env.SUPABASE_ANON_KEY ? 
                process.env.SUPABASE_ANON_KEY.startsWith('eyJ') : 
                false,
            length: process.env.SUPABASE_ANON_KEY ? process.env.SUPABASE_ANON_KEY.length : 0,
            preview: process.env.SUPABASE_ANON_KEY ? 
                process.env.SUPABASE_ANON_KEY.substring(0, 20) + '...' : 
                'NOT SET'
        }
    };

    // Test Supabase connection
    if (process.env.SUPABASE_URL && process.env.SUPABASE_ANON_KEY) {
        try {
            const { createClient } = await import('@supabase/supabase-js');
            const supabase = createClient(
                process.env.SUPABASE_URL,
                process.env.SUPABASE_ANON_KEY
            );

            response.tests.clientCreation = {
                success: true,
                message: 'Supabase client created successfully'
            };

            // Test database connection
            try {
                const { data, error } = await supabase
                    .from('user_preferences')
                    .select('*')
                    .limit(1);

                if (error) {
                    response.tests.databaseConnection = {
                        success: false,
                        error: error.message,
                        code: error.code,
                        details: error.details,
                        hint: error.hint
                    };
                } else {
                    response.tests.databaseConnection = {
                        success: true,
                        message: 'Database connection successful',
                        rowsReturned: data ? data.length : 0
                    };
                }
            } catch (dbError) {
                response.tests.databaseConnection = {
                    success: false,
                    error: dbError.message,
                    stack: dbError.stack
                };
            }
        } catch (clientError) {
            response.tests.clientCreation = {
                success: false,
                error: clientError.message,
                stack: clientError.stack
            };
        }
    } else {
        response.tests.clientCreation = {
            success: false,
            error: 'Missing environment variables'
        };
    }

    // Recommendations
    response.recommendations = [];

    if (!response.environment.SUPABASE_URL.exists) {
        response.recommendations.push('❌ Add SUPABASE_URL to Netlify environment variables');
    }

    if (!response.environment.SUPABASE_ANON_KEY.exists) {
        response.recommendations.push('❌ Add SUPABASE_ANON_KEY to Netlify environment variables');
    }

    if (response.environment.SUPABASE_ANON_KEY.exists && 
        !response.environment.SUPABASE_ANON_KEY.startsWithEyJ) {
        response.recommendations.push('⚠️ SUPABASE_ANON_KEY should start with "eyJ" - verify you copied the correct key');
    }

    if (response.environment.SUPABASE_ANON_KEY.length < 100) {
        response.recommendations.push('⚠️ SUPABASE_ANON_KEY seems too short - it should be 200+ characters');
    }

    if (response.tests.databaseConnection && !response.tests.databaseConnection.success) {
        const error = response.tests.databaseConnection;
        if (error.message && error.message.includes('API key')) {
            response.recommendations.push('❌ Invalid API key - Go to Supabase → Project Settings → API → Copy the "anon public" key again');
        }
        if (error.code === '42P01') {
            response.recommendations.push('❌ Table "user_preferences" does not exist - Run supabase-schema.sql in Supabase SQL Editor');
        }
    }

    if (response.recommendations.length === 0) {
        response.recommendations.push('✅ All checks passed! Your Supabase connection is working correctly.');
    }

    return {
        statusCode: 200,
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify(response, null, 2)
    };
};


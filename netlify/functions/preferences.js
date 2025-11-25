import { supabase, successResponse, errorResponse, handleSupabaseError } from './supabase-client.js';

export const handler = async (event) => {
    // Handle OPTIONS preflight request
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 204,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,POST,PUT,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        };
    }

    // Check if Supabase is configured
    if (!supabase) {
        return errorResponse(500, 'Database not configured');
    }

    const userId = 'default_user'; // You can add auth later

    try {
        // GET - Fetch user preferences
        if (event.httpMethod === 'GET') {
            const { data, error } = await supabase
                .from('user_preferences')
                .select('*')
                .eq('user_id', userId)
                .single();

            if (error) {
                // If no preferences exist yet, return defaults
                if (error.code === 'PGRST116') {
                    const today = new Date().toISOString().split('T')[0];
                    return successResponse({
                        user_id: userId,
                        start_date: today,
                        current_day: 1,
                        settings: {}
                    });
                }
                return handleSupabaseError(error);
            }

            return successResponse(data);
        }

        // POST/PUT - Create or update user preferences
        if (event.httpMethod === 'POST' || event.httpMethod === 'PUT') {
            const payload = JSON.parse(event.body || '{}');
            const { startDate, currentDay, settings } = payload;

            // Prepare update data
            const updateData = {
                user_id: userId
            };

            if (startDate !== undefined) {
                updateData.start_date = startDate;
            }
            if (currentDay !== undefined) {
                updateData.current_day = parseInt(currentDay);
            }
            if (settings !== undefined) {
                updateData.settings = settings;
            }

            // Upsert (insert or update)
            const { data, error } = await supabase
                .from('user_preferences')
                .upsert(updateData, {
                    onConflict: 'user_id'
                })
                .select()
                .single();

            if (error) return handleSupabaseError(error);

            return successResponse(data);
        }

        return errorResponse(405, 'Method not allowed');

    } catch (error) {
        console.error('Preferences function error:', error);
        return errorResponse(500, 'Internal server error');
    }
};


import { createClient } from '@supabase/supabase-js';

export default defineComponent({
  async run({ steps, $ }) {
    // Configuration variables
    const TOP_N_ARTICLES = 5; // Default number of articles
    const CALLBACK_URL = "https://eokxolvztxbzrb9.m.pipedream.net"; // Set your callback URL here
    const API_KEY = "4b9e132b4145b86bcce9adae2a4f4f2cd8398d6a3f022a7dc9a03711e6167d52";
    const ENDPOINT_URL = "https://webhook.site/e1a87268-271e-4e95-b93e-35c3466b1693";
    const DELAY_MS = 1000; // 500ms delay between requests
    
    // Supabase configuration - UPDATE THESE WITH YOUR ACTUAL VALUES
    const SUPABASE_URL = process.env.SUPABASE_URL; // Set this in your Pipedream environment variables
    const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY; // Set this in your Pipedream environment variables
    
    // Initialize Supabase client
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    
    // Get current date in YYYY-MM-DD format
    const getCurrentDate = () => {
      const today = new Date();
      return today.toISOString().split('T')[0];
    };
    
    // Sleep function for rate limiting
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    
    // Get users data from previous step
    const users = steps.get_subscribed_users.$return_value.data;
    
    if (!users || !Array.isArray(users)) {
      throw new Error("No users data found or data is not an array");
    }
    
    console.log(`Processing ${users.length} users`);
    
    // Process each user
    for (let i = 0; i < users.length; i++) {
      const user = users[i];
      
      try {
        // Prepare payload with user info and defaults for missing fields
        const payload = {
          user_info: {
            name: user.name || "Default User",
            title: user.title || "User",
            goals: user.goals || "General learning and development"
          },
          target_date: getCurrentDate(),
          top_n_articles: TOP_N_ARTICLES,
          callback_url: CALLBACK_URL
        };
        
        console.log(`Sending request for user: ${payload.user_info.name} (${user.email})`);
        
        // Make the API request and wait for response
        const response = await fetch(ENDPOINT_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY
          },
          body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
          throw new Error(`API request failed with status: ${response.status}`);
        }
        
        const responseData = await response.json();
        console.log(`✓ Request sent successfully for ${payload.user_info.name}, task_id: ${responseData.task_id}`);
        
        // Update Supabase profiles table with task_id
        if (responseData.task_id) {
          const { data, error } = await supabase
            .from('profiles')
            .update({ task_id: responseData.task_id })
            .eq('email', user.email)
            .select();
          
          if (error) {
            console.error(`✗ Failed to update Supabase for ${user.email}:`, error.message);
          } else {
            console.log(`✓ Updated Supabase task_id for ${user.email}`);
          }
        }
        
        // Add delay between requests (except for the last one)
        if (i < users.length - 1) {
          await sleep(DELAY_MS);
        }
        
      } catch (error) {
        console.error(`✗ Failed to send request for user ${user.name || user.email}:`, error.message);
        // Continue processing other users even if one fails
      }
    }
    
    console.log(`Completed processing all ${users.length} users`);
    
    // Return summary
    return {
      message: `Processed ${users.length} users`,
      users_processed: users.length,
      target_date: getCurrentDate(),
      top_n_articles: TOP_N_ARTICLES
    };
  },
});
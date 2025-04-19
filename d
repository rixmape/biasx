[33mcommit d8874f28a206d9288eb93ddfb7d45475187fcead[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mimproved-docs[m[33m, [m[1;31morigin/improved-docs[m[33m)[m
Merge: f875bbc 901e29d
Author: Jerwin Glen Lucero <jerwinglenalejandre.lucero@bicol-u.edu.ph>
Date:   Sat Apr 19 11:22:19 2025 +0800

    add logo in the header

[33mcommit f875bbc9f3f60285f27f7ae44ba60ab39108fca2[m
Author: Jerwin Glen Lucero <jerwinglenalejandre.lucero@bicol-u.edu.ph>
Date:   Sat Apr 19 11:19:54 2025 +0800

    add logo in the header

[33mcommit 901e29deced002b344b0ace408d20d37ada5c39f[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 00:00:31 2025 +0800

    Refactor documentation and styling for improved clarity and consistency

[33mcommit f9fc0a69024af2fc43cc0eb404d9f424c7b33da4[m
Author: Jerwin Glen Lucero <jerwinglenalejandre.lucero@bicol-u.edu.ph>
Date:   Wed Apr 16 09:06:39 2025 +0800

    Improve Docs

[33mcommit 04e692215ed2d21300c40f8ceaf01db62d1d1db7[m
Author: Jerwin Glen Lucero <jerwinglenalejandre.lucero@bicol-u.edu.ph>
Date:   Wed Apr 9 12:55:40 2025 +0800

    Improve Documentation Page

[33mcommit a8cabecfd9f40c8de0460ee5a7b2d688c7a59e6b[m
Merge: 80947ed c254628
Author: Rix Mape <88783714+rixmape@users.noreply.github.com>
Date:   Sat Apr 19 10:38:31 2025 +0800

    Merge pull request #21 from rixmape/expander-config
    
    Enhance configuration page and add home page with descriptions

[33mcommit c2546286da27ef1057e9394ba573d6ebf2d433ba[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 10:31:30 2025 +0800

    Add configuration containers

[33mcommit 7ebe36566d7d06715fc8aca307a092a7f71ca1d4[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 10:01:10 2025 +0800

    Add home page and chart descriptions; update session state management
    
    - Implement home page with introductory content and usage instructions.
    - Add chart descriptions for various metrics in a new YAML file.
    - Update session state handling to control visibility of upload page.
    - Refactor file paths for loading chart descriptions.

[33mcommit 3808333825086c4ec18618e796c7ced1b81c29b4[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 08:53:40 2025 +0800

    Add chart descriptions

[33mcommit c9856e1bc3c0b006885cbddc6b2cb42a1983c576[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 08:36:33 2025 +0800

    Add whitespace separator between input sections

[33mcommit 15860f9155b4f6573ba13e2bbf84ab97f6ed844f[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 08:28:55 2025 +0800

    Add help messages in config page

[33mcommit 30f48a5bd943819c73504aca88a604546d6cd64b[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 08:06:27 2025 +0800

    Remove section numbers in config page

[33mcommit c96fd013e5f532aad5ca7fffe7103163badbc117[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 19 00:00:31 2025 +0800

    Refactor documentation and styling for improved clarity and consistency

[33mcommit 99124ad9cf81565471417eeeeabc3d82caea4a76[m
Author: Jerwin Glen Lucero <jerwinglenalejandre.lucero@bicol-u.edu.ph>
Date:   Wed Apr 16 09:06:39 2025 +0800

    Improve Docs

[33mcommit df9c8b4442b751d3cb68903c45b1c2e85d5c9e36[m
Author: Jerwin Glen Lucero <jerwinglenalejandre.lucero@bicol-u.edu.ph>
Date:   Wed Apr 9 12:55:40 2025 +0800

    Improve Documentation Page

[33mcommit 80947eda771798b3043bf6c379d83227b9ed2721[m
Merge: c782204 add9f5c
Author: Rix Mape <88783714+rixmape@users.noreply.github.com>
Date:   Fri Apr 18 21:47:39 2025 +0800

    Merge pull request #20 from rixmape/cleaner-interface
    
    Refactor analysis and visualization structure

[33mcommit add9f5c6fcdd7b880d1c9fc17d1fcb33dbc002d4[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Fri Apr 18 21:43:08 2025 +0800

    Refactor analysis flow and session state management

[33mcommit 6f60f7fb782a3ae0d3e918a5faeeaf19c80c1bc0[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Fri Apr 18 21:27:08 2025 +0800

    Add image details dialog

[33mcommit 9842b7517ed2f94e176ad7aaa540cc76147b2c47[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Fri Apr 18 20:51:39 2025 +0800

    Refactor visualizations and configuration structure
    
    - Removed the graphs.py file and split its functionality into dedicated modules for feature analysis, model performance, and image analysis.
    - Introduced config.py to manage model and dataset configurations in a structured manner.
    - Added results.py to handle the display of analysis results across different tabs.
    - Created upload.py for model selection and upload functionality.
    - Updated utils.py to streamline utility functions and improve session state management.
    - Enhanced visualizations with improved layout and styling for better user experience.

[33mcommit 21ae7e1d9a608c93858489a0a05f3e5122175ff5[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Fri Apr 18 17:09:02 2025 +0800

    Fix library configuration

[33mcommit c7822041084734fe9d81c305961ff876c16dee1d[m
Merge: c092337 8a8b150
Author: Rix Mape <88783714+rixmape@users.noreply.github.com>
Date:   Fri Apr 18 16:55:48 2025 +0800

    Merge pull request #19 from rixmape/with-male-proportion
    
    Enhance experimental framework with visualizations and logging improvements

[33mcommit 8a8b1501ce73861649402534ac8bccde3e72f15c[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sat Apr 5 15:13:47 2025 +0800

    Update to comment warning messages

[33mcommit ae59263fd940273863170c98c52febed25d7b6f8[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Fri Apr 4 00:41:59 2025 +0800

    Update to save age and race

[33mcommit 36beb696069c25075bcae2eff6cb3e726356a4d2[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Wed Apr 2 23:35:49 2025 +0800

    Add more visualizations

[33mcommit 90cc8577bb23957a9dede6b846c062db4535b225[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Wed Apr 2 21:17:43 2025 +0800

    Add visualizations for one experiment

[33mcommit 5798a7084bbdcc17c5ba00245c8b807ee044942b[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Wed Apr 2 20:10:29 2025 +0800

    Update JSON schema

[33mcommit 7526e08c8235a970d51be4c4cc3246b2974c6e95[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Wed Apr 2 19:56:02 2025 +0800

    Update to centralize logging

[33mcommit 135ddfd06fa058a006f74a458d6e6703274edc6a[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Wed Apr 2 12:48:28 2025 +0800

    Update to run single experiment only

[33mcommit 03d1b27388d05e0ad0b5cea00cc1cedeefa65f50[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Tue Apr 1 13:18:01 2025 +0800

    Add explanations to experiment results

[33mcommit e3fdf39fa5794581aeb476624d72c21a4bd5f38c[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Mon Mar 31 20:55:46 2025 +0800

    Add dataset saving feature

[33mcommit 752dfc0a9b33095ca680f787a6e01f023e8b65fa[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Mon Mar 31 19:38:52 2025 +0800

    Fix landmark detection for RGB training images

[33mcommit 6578c0671b1d2f5e80156fd51e1e98bb6859e0a9[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Mon Mar 31 17:05:51 2025 +0800

    Update to mask multiple features per image

[33mcommit 61dd818095d088871e52d6e0790ce458eecad25d[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Mon Mar 31 16:12:39 2025 +0800

    Add data validation for config options

[33mcommit 6fab96a9804f757bd1f8a9de04e2380083084ffc[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Mon Mar 31 15:47:35 2025 +0800

    Update formula for equalized odds

[33mcommit 7a07c3940ca6c0bf2386bc24bf16c51306cee9f5[m
Author: rixmape <rixdonninorecario.mape@bicol-u.edu.ph>
Date:   Sun Mar 30 21:51:49 2025 +0800

    Update log messages for readability

[33mcommit 33fea5
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def statewise_stores_plot(train_df_copy):
    '''
        plotting the statewise number of stores
        args:
            train_df_copy = dataframe copy of train_df
    '''
    d = train_df_copy.groupby(['state_id', 'store_id']).count().reset_index()
    stores_in_states = d['state_id'].value_counts()
    colors = plt.cm.viridis(np.linspace(0, 1, len(stores_in_states)))

    ax = stores_in_states.plot(kind="bar", color=colors, title="Number of Stores by State", yticks=[0,1,2,3,4])
    plt.xticks(rotation=0)

    for i, count in enumerate(stores_in_states):
        ax.text(i, count + 0.01, str(count), ha='center', va='bottom', fontsize=10)

    plt.show()


def statewise_items_plot(train_df_copy):
    # Group by 'state_id' and 'item_id', then count occurrences
    d = train_df_copy.groupby(['state_id', 'item_id']).size().reset_index(name='count')

    # Count how many unique 'item_id's per 'state_id'
    stores_in_states = d['state_id'].value_counts()

    # Generate colors for the bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(stores_in_states)))

    # Create the bar plot
    ax = stores_in_states.plot(kind="bar", color=colors, title="Number of Unique Items by State")
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal

    # Annotate each bar with the corresponding count
    for i, count in enumerate(stores_in_states):
        ax.text(i, count + 0.01, str(count), ha='center', va='bottom', fontsize=10)

    # Display the plot
    plt.show()


def statewise_category_total_sales_plot(train_df_copy):
    states_to_plot = ['CA', 'TX', 'WI']

    for state in states_to_plot:
        # Filter data for the specific state
        state_df = train_df_copy[train_df_copy['state_id'] == state]
        
        # Group by 'date', 'category', and 'store_id', then sum the sales for each combination
        state_category_store_sales = state_df.groupby(['date', 'category', 'state_id'])['total_sales'].sum().reset_index()
        
        # Pivot the DataFrame to have 'date' as index, 'category' and 'store_id' as columns, and sales as values
        sales_pivot = state_category_store_sales.pivot_table(index='date', columns=['state_id', 'category'], values='total_sales', aggfunc='sum')
        
        # Plot the total sales over time, categorized by store and category for the specific state
        ax = sales_pivot.plot(kind='line', figsize=(12, 6), title=f"Total Sales Over Time by Store and Category for {state}")
        
        # Customize the plot
        ax.set_ylabel("Total Sales")
        ax.set_xlabel("Date")
        plt.xticks(rotation=45, ha='right')
        
        # Show the plot for this state
        plt.tight_layout()
        plt.show()


def storeid_category_total_sales_plot(train_df_copy):
    total_sales_by_state_category = train_df_copy.groupby(['store_id', 'category'])['total_sales'].sum().reset_index()

    # Create a box plot to visualize total sales distribution for each state by category
    plt.figure(figsize=(12, 6))

    # Boxplot for total sales by state and category
    sns.barplot(x='store_id', y='total_sales', hue='category', data=total_sales_by_state_category)

    # Add title and labels
    plt.title("Plot of Total Sales by Store Id and Category")
    plt.ylabel("Total Sales")
    plt.xlabel("Store ID")

    # Display the plot
    plt.tight_layout()
    plt.show()
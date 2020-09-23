# 2020_data_mining_assignments
[link to assignment](http://www.cs.uu.nl/docs/vakken/mdm/assignment1-2020.pdf)

## Part1, tree algorithm/implementation:
- [X] tree_grow functie die het [pseudocode in de slides](./media/tree_grow_pseudo_code.png) volgt
- [ ] tree_grow aanpassen voor n_feat, een paar lines die zeggen dat de rows die aan exhaustivesplitsearch gegeven worden random uit x gepakt moeten worden
- [ ] tree_grow_b bootstrap versie van tree grow, die een lijst van tree construct door met replacement rows uit x te kiezen

- [X] [tree_pred functie](https://github.com/Vinkage/2020_data_mining_assignments/blob/da8ca975fb9d11d3801fef66344736e675734c42/assignment1.py#L77-L103) met efficiente conditional branches 
- [ ] tree_pred_b een functie die een lijst van tree kan gebruiken om een voorspelling te maken voor rows in een data array x

## Part2, data analysis:
- [ ] Datasets collecten uit de literature
- [ ] Datasets describen, exploren/plotten/formatten als het nodig is
- [ ] Figure out how to compute accuracy, precision, recall for predictions in python
- [ ] Figure out how we want to compute the confusion matrix (scipy?)

### Official steps
![](./steps_data_anal.png)

## The report
![](./report_reqs.png)

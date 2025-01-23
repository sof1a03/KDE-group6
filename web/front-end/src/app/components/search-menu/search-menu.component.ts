import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { UserService } from '../../user.service';
import { Router } from '@angular/router';
import { CategoryService } from '../../category.service';
import { OnInit } from '@angular/core';

@Component({
  selector: 'app-search-menu',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './search-menu.component.html',
  styleUrl: './search-menu.component.css'
})
export class SearchMenuComponent implements OnInit {
  categories: string[] = [];
  selectedCategories = [];

  startYear = 1900;
  endYear = new Date().getFullYear();
  years: number[] = [];

  constructor(public userService: UserService,
    private router: Router,
    private categoryService: CategoryService
  ) {
    for (let year = this.endYear; year >= this.startYear; year--) {
      this.years.push(year);
    }
  }

  ngOnInit() {
    this.categoryService.getCategories().subscribe(categories => {
      console.log(categories)
      this.categories = categories;
    })
  }
  onSubmit() {
    const isbn = (<HTMLInputElement>document.getElementById('isbn')).value;
    const title = (<HTMLInputElement>document.getElementById('title')).value;
    const author = (<HTMLInputElement>document.getElementById('author')).value;
    const publisher = (<HTMLInputElement>document.getElementById('publisher')).value;
    const startYear = (<HTMLSelectElement>document.getElementById('startYear')).value;
    const endYear = (<HTMLSelectElement>document.getElementById('endYear')).value;

    this.router.navigate(['/search'], {
      queryParams: {
        categories: this.selectedCategories,
        isbn: isbn,
        title: title,
        author: author,
        publisher: publisher,
        startYear: startYear,
        endYear: endYear
      }
    });
  }
}
